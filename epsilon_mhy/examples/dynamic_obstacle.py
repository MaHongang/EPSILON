#!/usr/bin/env python3
"""Example of SSC planner with dynamic obstacles.

This example demonstrates trajectory planning with a dynamic obstacle
that crosses the ego vehicle's path.
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from epsilon_mhy.core.state import State
from epsilon_mhy.core.lane import Lane
from epsilon_mhy.core.vehicle import VehicleParam, ObstacleTrajectory
from epsilon_mhy.planning.ssc_planner import SscPlanner, SscPlannerConfig, create_straight_lane


def create_moving_obstacle(
    start_position: np.ndarray,
    velocity: np.ndarray,
    start_time: float = 0.0,
    duration: float = 10.0,
    dt: float = 0.1
) -> ObstacleTrajectory:
    """Create an obstacle moving with constant velocity.
    
    Args:
        start_position: (x, y) start position
        velocity: (vx, vy) velocity
        start_time: Start time
        duration: Duration of trajectory
        dt: Time step
        
    Returns:
        ObstacleTrajectory
    """
    states = []
    t = start_time
    pos = start_position.copy()
    
    while t <= start_time + duration:
        angle = np.arctan2(velocity[1], velocity[0])
        speed = np.linalg.norm(velocity)
        
        states.append(State(
            time_stamp=t,
            position=pos.copy(),
            angle=angle,
            velocity=speed
        ))
        
        pos += velocity * dt
        t += dt
    
    return ObstacleTrajectory.from_state_list(
        states=states,
        vehicle_param=VehicleParam(
            width=1.8,
            length=4.5,
            d_cr=1.2
        )
    )


def main():
    print("=" * 60)
    print("SSC Planner - Dynamic Obstacle Example")
    print("=" * 60)
    
    # Create a straight reference lane
    lane = create_straight_lane(length=200.0, num_points=100)
    print(f"Created straight lane: length = {lane.length:.1f}m")
    
    # Create initial ego state
    initial_state = State(
        time_stamp=0.0,
        position=np.array([10.0, 0.0]),
        angle=0.0,
        velocity=10.0,
        acceleration=0.0
    )
    print(f"Ego initial state: pos=({initial_state.x:.1f}, {initial_state.y:.1f}), "
          f"v={initial_state.velocity:.1f} m/s")
    
    # Create a dynamic obstacle that will be ahead of ego
    # Obstacle starts ahead and moves slower
    obstacle1 = create_moving_obstacle(
        start_position=np.array([50.0, 0.0]),  # 40m ahead of ego
        velocity=np.array([5.0, 0.0]),  # Moving at 5 m/s (slower than ego)
        start_time=0.0,
        duration=10.0
    )
    print(f"Obstacle 1: starts at x=50m, velocity=5 m/s (slower vehicle ahead)")
    
    # Create another obstacle that will cross the path
    obstacle2 = create_moving_obstacle(
        start_position=np.array([70.0, 10.0]),  # Starts to the side
        velocity=np.array([0.0, -3.0]),  # Moving across the road
        start_time=0.0,
        duration=10.0
    )
    print(f"Obstacle 2: starts at (70, 10), crossing the road")
    
    obstacles = [obstacle1, obstacle2]
    
    # Create planner configuration
    config = SscPlannerConfig(
        planning_horizon=8.0,
        target_velocity=12.0,
        weight_proximity=1.0
    )
    
    planner = SscPlanner(config)
    
    # Plan trajectory
    print("\nPlanning trajectory with dynamic obstacles...")
    result = planner.plan(
        initial_state=initial_state,
        reference_lane=lane,
        obstacle_trajectories=obstacles,
        target_velocity=12.0
    )
    
    print(f"\nPlanning result:")
    print(f"  Success: {result.success}")
    print(f"  Message: {result.message}")
    print(f"  Time cost: {result.time_cost:.2f} ms")
    
    if result.success:
        print(f"  Number of trajectory points: {len(result.cartesian_states)}")
        
        # Analyze the trajectory
        print("\nTrajectory analysis:")
        
        # Find minimum distance to obstacles
        min_distances = []
        for t_idx, state in enumerate(result.cartesian_states):
            t = state.time_stamp
            for obs in obstacles:
                obs_state = obs.get_state_at_time(t)
                if obs_state:
                    dist = np.linalg.norm(state.position - obs_state.position)
                    min_distances.append((t, dist))
        
        if min_distances:
            min_dist_time, min_dist = min(min_distances, key=lambda x: x[1])
            print(f"  Minimum distance to obstacles: {min_dist:.2f}m at t={min_dist_time:.2f}s")
        
        # Velocity profile
        velocities = [s.velocity for s in result.cartesian_states]
        print(f"  Velocity range: [{min(velocities):.2f}, {max(velocities):.2f}] m/s")
        
        # Lateral deviation
        if result.frenet_states:
            d_vals = [fs.vec_dt[0] for fs in result.frenet_states]
            print(f"  Lateral deviation range: [{min(d_vals):.2f}, {max(d_vals):.2f}] m")
        
        # Print key trajectory points
        print("\nKey trajectory points:")
        times_to_show = [0.0, 2.0, 4.0, 6.0, 8.0]
        for target_t in times_to_show:
            for s in result.cartesian_states:
                if abs(s.time_stamp - target_t) < 0.05:
                    print(f"  t={s.time_stamp:.1f}s: pos=({s.x:.1f}, {s.y:.1f}), "
                          f"v={s.velocity:.1f} m/s")
                    break
    else:
        print("\nPlanning failed. This might be because:")
        print("  - The corridor generation couldn't find a free path")
        print("  - The QP optimization couldn't find a feasible solution")
        print("  - Try adjusting obstacle positions or planner parameters")
    
    # Visualization
    try:
        from epsilon_mhy.utils.visualization import Visualizer
        import matplotlib.pyplot as plt
        
        print("\nGenerating visualization...")
        vis = Visualizer()
        
        # Plot scenario
        fig1 = vis.plot_scenario(
            lane=lane,
            ego_state=initial_state,
            obstacles=obstacles,
            trajectory_states=result.cartesian_states if result.success else None,
            title="Dynamic Obstacle Avoidance"
        )
        
        # Plot Frenet trajectory if successful
        if result.success and result.frenet_states:
            fig2 = vis.plot_frenet_trajectory(
                frenet_states=result.frenet_states,
                title="Planned Trajectory in Frenet Frame"
            )
        
        plt.show()
        
    except ImportError:
        print("\nVisualization skipped (matplotlib not available)")
    
    return result.success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
