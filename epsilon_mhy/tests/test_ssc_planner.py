#!/usr/bin/env python3
"""Unit tests for SSC planner components."""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class TestState(unittest.TestCase):
    """Tests for State class."""
    
    def test_state_creation(self):
        from epsilon_mhy.core.state import State
        
        state = State(
            time_stamp=1.0,
            position=np.array([10.0, 5.0]),
            angle=0.5,
            velocity=10.0
        )
        
        self.assertEqual(state.time_stamp, 1.0)
        self.assertEqual(state.x, 10.0)
        self.assertEqual(state.y, 5.0)
        self.assertEqual(state.angle, 0.5)
        self.assertEqual(state.velocity, 10.0)
    
    def test_state_copy(self):
        from epsilon_mhy.core.state import State
        
        state = State(position=np.array([1.0, 2.0]), velocity=5.0)
        state_copy = state.copy()
        
        # Modify original
        state.position[0] = 100.0
        
        # Copy should be unchanged
        self.assertEqual(state_copy.position[0], 1.0)


class TestFrenetState(unittest.TestCase):
    """Tests for FrenetState class."""
    
    def test_frenet_state_from_s_dt(self):
        from epsilon_mhy.core.state import FrenetState
        
        fs = FrenetState.from_s_dt(
            s=np.array([10.0, 5.0, 0.0]),
            dt=np.array([1.0, 0.5, 0.0]),
            time_stamp=0.0
        )
        
        self.assertEqual(fs.s, 10.0)
        self.assertEqual(fs.s_dot, 5.0)
        self.assertEqual(fs.d, 1.0)
        self.assertEqual(fs.d_dot, 0.5)
    
    def test_frenet_ds_computation(self):
        from epsilon_mhy.core.state import FrenetState
        
        # With non-zero s_dot, ds should be computed
        fs = FrenetState.from_s_dt(
            s=np.array([10.0, 5.0, 1.0]),
            dt=np.array([1.0, 2.5, 0.0]),
            time_stamp=0.0
        )
        
        # d' = dd/dt / ds/dt = 2.5 / 5.0 = 0.5
        self.assertAlmostEqual(fs.vec_ds[1], 0.5, places=5)
        self.assertTrue(fs.is_ds_usable)


class TestLane(unittest.TestCase):
    """Tests for Lane class."""
    
    def test_straight_lane(self):
        from epsilon_mhy.core.lane import Lane
        
        points = np.array([[0, 0], [10, 0], [20, 0], [30, 0]])
        lane = Lane.from_points(points)
        
        self.assertAlmostEqual(lane.length, 30.0, places=2)
        
        # Position at midpoint
        pos = lane.get_position(15.0)
        self.assertAlmostEqual(pos[0], 15.0, places=1)
        self.assertAlmostEqual(pos[1], 0.0, places=1)
    
    def test_lane_tangent(self):
        from epsilon_mhy.core.lane import Lane
        
        # Straight lane along x-axis
        points = np.array([[0, 0], [10, 0], [20, 0]])
        lane = Lane.from_points(points)
        
        tangent = lane.get_tangent(5.0)
        self.assertAlmostEqual(tangent[0], 1.0, places=2)
        self.assertAlmostEqual(tangent[1], 0.0, places=2)
    
    def test_frenet_point_conversion(self):
        from epsilon_mhy.core.lane import Lane
        
        points = np.array([[0, 0], [10, 0], [20, 0], [30, 0]])
        lane = Lane.from_points(points)
        
        # Point on the lane
        s, d = lane.get_frenet_point(np.array([15.0, 0.0]))
        self.assertAlmostEqual(s, 15.0, places=1)
        self.assertAlmostEqual(d, 0.0, places=1)
        
        # Point off the lane (left)
        s, d = lane.get_frenet_point(np.array([15.0, 2.0]))
        self.assertAlmostEqual(s, 15.0, places=1)
        self.assertAlmostEqual(d, 2.0, places=1)


class TestVehicle(unittest.TestCase):
    """Tests for Vehicle class."""
    
    def test_vehicle_vertices(self):
        from epsilon_mhy.core.state import State
        from epsilon_mhy.core.vehicle import VehicleParam
        
        param = VehicleParam(
            width=2.0,
            length=4.0,
            rear_suspension=1.0
        )
        
        state = State(position=np.array([0.0, 0.0]), angle=0.0)
        vertices = param.get_vertices(state)
        
        self.assertEqual(vertices.shape, (4, 2))
        
        # Check that vertices form a rectangle
        # Front should be at x > 0, rear at x < 0 (for angle=0)
        self.assertTrue(np.all(vertices[:2, 0] > 0))  # Front corners
        self.assertTrue(np.all(vertices[2:, 0] < 0))  # Rear corners


class TestBezier(unittest.TestCase):
    """Tests for Bezier curve."""
    
    def test_bezier_curve_endpoints(self):
        from epsilon_mhy.math.bezier import BezierCurve
        
        control_points = np.array([
            [0.0, 0.0],
            [1.0, 2.0],
            [3.0, 3.0],
            [4.0, 0.0]
        ])
        
        curve = BezierCurve(control_points, t_start=0.0, t_end=1.0)
        
        # Bezier curve passes through first and last control points
        start = curve.evaluate(0.0)
        end = curve.evaluate(1.0)
        
        np.testing.assert_array_almost_equal(start, [0.0, 0.0])
        np.testing.assert_array_almost_equal(end, [4.0, 0.0])
    
    def test_bezier_spline(self):
        from epsilon_mhy.math.bezier import BezierSpline
        
        spline = BezierSpline(degree=3, dimension=2)
        
        # Add two segments
        pts1 = np.array([[0, 0], [1, 1], [2, 1], [3, 0]])
        pts2 = np.array([[3, 0], [4, -1], [5, -1], [6, 0]])
        
        spline.add_segment(pts1, 0.0, 1.0)
        spline.add_segment(pts2, 1.0, 2.0)
        
        self.assertEqual(spline.num_segments, 2)
        self.assertEqual(spline.begin, 0.0)
        self.assertEqual(spline.end, 2.0)


class TestStateTransformer(unittest.TestCase):
    """Tests for Frenet coordinate transformation."""
    
    def test_global_to_frenet_on_lane(self):
        from epsilon_mhy.core.state import State
        from epsilon_mhy.core.lane import Lane
        from epsilon_mhy.math.frenet import StateTransformer
        
        # Straight lane
        points = np.array([[0, 0], [50, 0], [100, 0]])
        lane = Lane.from_points(points)
        transformer = StateTransformer(lane)
        
        # State on the lane
        state = State(
            position=np.array([25.0, 0.0]),
            angle=0.0,
            velocity=10.0
        )
        
        fs = transformer.global_to_frenet(state)
        
        self.assertAlmostEqual(fs.s, 25.0, places=1)
        self.assertAlmostEqual(fs.d, 0.0, places=1)
        self.assertAlmostEqual(fs.s_dot, 10.0, places=1)


class TestSscMap(unittest.TestCase):
    """Tests for SSC map."""
    
    def test_map_creation(self):
        from epsilon_mhy.core.state import FrenetState
        from epsilon_mhy.planning.ssc_map import SscMap, SscMapConfig
        
        config = SscMapConfig(
            map_size=(100, 20, 50),
            map_resolution=(0.5, 0.2, 0.1)
        )
        
        ssc_map = SscMap(config)
        
        initial_fs = FrenetState.from_s_dt(
            s=np.array([10.0, 5.0, 0.0]),
            dt=np.array([0.0, 0.0, 0.0]),
            time_stamp=0.0
        )
        
        ssc_map.reset(initial_fs)
        
        self.assertEqual(ssc_map.grid.shape, (100, 20, 50))
        self.assertEqual(ssc_map.start_time, 0.0)


class TestCorridorGenerator(unittest.TestCase):
    """Tests for corridor generation."""
    
    def test_corridor_generation_free_space(self):
        from epsilon_mhy.core.state import FrenetState
        from epsilon_mhy.planning.ssc_map import SscMap, SscMapConfig
        from epsilon_mhy.planning.corridor import CorridorGenerator, generate_initial_trajectory_fs
        
        config = SscMapConfig(
            map_size=(100, 20, 50),
            map_resolution=(0.5, 0.2, 0.1)
        )
        
        ssc_map = SscMap(config)
        
        initial_fs = FrenetState.from_s_dt(
            s=np.array([10.0, 5.0, 0.0]),
            dt=np.array([0.0, 0.0, 0.0]),
            time_stamp=0.0
        )
        
        ssc_map.reset(initial_fs)
        
        # Generate initial trajectory
        traj_fs = generate_initial_trajectory_fs(
            initial_fs=initial_fs,
            target_velocity=5.0,
            duration=3.0,
            dt=0.2
        )
        
        # Generate corridor
        generator = CorridorGenerator(ssc_map, config)
        corridor = generator.generate_corridor(traj_fs)
        
        # In free space, corridor should be valid
        self.assertTrue(corridor.is_valid)
        self.assertGreater(len(corridor.cubes), 0)


class TestSscPlanner(unittest.TestCase):
    """Integration tests for SSC planner."""
    
    def test_planner_no_obstacles(self):
        from epsilon_mhy.core.state import State
        from epsilon_mhy.planning.ssc_planner import SscPlanner, SscPlannerConfig, create_straight_lane
        
        lane = create_straight_lane(length=100.0)
        
        initial_state = State(
            time_stamp=0.0,
            position=np.array([5.0, 0.0]),
            angle=0.0,
            velocity=10.0
        )
        
        config = SscPlannerConfig(
            planning_horizon=3.0,
            target_velocity=10.0
        )
        
        planner = SscPlanner(config)
        
        result = planner.plan(
            initial_state=initial_state,
            reference_lane=lane,
            obstacle_trajectories=[]
        )
        
        # Planning should succeed with no obstacles
        self.assertTrue(result.success, f"Planning failed: {result.message}")
        self.assertGreater(len(result.frenet_states), 0)
        self.assertGreater(len(result.cartesian_states), 0)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestState))
    suite.addTests(loader.loadTestsFromTestCase(TestFrenetState))
    suite.addTests(loader.loadTestsFromTestCase(TestLane))
    suite.addTests(loader.loadTestsFromTestCase(TestVehicle))
    suite.addTests(loader.loadTestsFromTestCase(TestBezier))
    suite.addTests(loader.loadTestsFromTestCase(TestStateTransformer))
    suite.addTests(loader.loadTestsFromTestCase(TestSscMap))
    suite.addTests(loader.loadTestsFromTestCase(TestCorridorGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestSscPlanner))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
