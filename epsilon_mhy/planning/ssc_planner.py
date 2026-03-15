"""SSC (Spatio-temporal Semantic Corridor) Planner.

Reference: EPSILON/util/ssc_planner/src/ssc_planner/ssc_planner.cc

The SSC planner generates safe trajectories for autonomous vehicles by:
1. Building a 3D spatio-temporal map in Frenet coordinates
2. Filling the map with obstacle occupancy
3. Generating safe corridors using seed-based inflation
4. Optimizing a Bezier spline trajectory within the corridors
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

from ..core.state import State, FrenetState
from ..core.lane import Lane
from ..core.vehicle import Vehicle, VehicleParam, ObstacleTrajectory, FsVehicle
from ..core.types import ErrorType
from ..math.frenet import StateTransformer
from ..math.bezier import BezierSpline
from ..math.qp_solver import optimize_bezier_in_corridor, Corridor
from .ssc_map import SscMap, SscMapConfig
from .corridor import CorridorGenerator, DrivingCorridor, generate_initial_trajectory_fs


@dataclass
class SscPlannerConfig:
    """Configuration for SSC Planner.
    
    Attributes:
        map_config: SSC map configuration
        weight_proximity: Weight for reference point proximity in optimization
        low_speed_threshold: Threshold for switching to low-speed mode
        planning_horizon: Planning time horizon in seconds
        planning_dt: Time step for trajectory sampling
        target_velocity: Default target velocity
    """
    map_config: SscMapConfig = field(default_factory=SscMapConfig)
    weight_proximity: float = 1.0
    low_speed_threshold: float = 2.0
    planning_horizon: float = 8.0
    planning_dt: float = 0.1
    target_velocity: float = 10.0
    bezier_degree: int = 5


@dataclass
class PlanningResult:
    """Result of planning operation.
    
    Attributes:
        success: Whether planning succeeded
        trajectory: Optimized trajectory (BezierSpline in Frenet frame)
        corridor: Generated driving corridor
        frenet_states: Sampled states in Frenet frame
        cartesian_states: Sampled states in Cartesian frame
        time_cost: Computation time in milliseconds
        message: Status message
    """
    success: bool = False
    trajectory: Optional[BezierSpline] = None
    corridor: Optional[DrivingCorridor] = None
    frenet_states: List[FrenetState] = field(default_factory=list)
    cartesian_states: List[State] = field(default_factory=list)
    time_cost: float = 0.0
    message: str = ""


class SscPlanner:
    """Spatio-temporal Semantic Corridor Planner.
    
    Main planner class that integrates all components for trajectory planning.
    
    Usage:
        planner = SscPlanner(config)
        result = planner.plan(
            initial_state=state,
            reference_lane=lane,
            obstacle_trajectories=obstacles
        )
    """
    
    def __init__(self, config: Optional[SscPlannerConfig] = None):
        """Initialize SSC planner.
        
        Args:
            config: Planner configuration
        """
        self.config = config or SscPlannerConfig()
        
        # Initialize SSC map
        self.ssc_map = SscMap(self.config.map_config)
        
        # Initialize corridor generator
        self.corridor_generator = CorridorGenerator(self.ssc_map)
        
        # State transformer (set when reference lane is provided)
        self.state_transformer: Optional[StateTransformer] = None
        
        # Cached data
        self.reference_lane: Optional[Lane] = None
        self.initial_frenet_state: Optional[FrenetState] = None
    
    def plan(self,
             initial_state: State,
             reference_lane: Lane,
             obstacle_trajectories: List[ObstacleTrajectory],
             target_velocity: Optional[float] = None,
             ego_param: Optional[VehicleParam] = None) -> PlanningResult:
        """Plan a trajectory avoiding dynamic obstacles.
        
        Args:
            initial_state: Current ego vehicle state
            reference_lane: Reference lane to follow
            obstacle_trajectories: List of dynamic obstacle trajectories
            target_velocity: Target velocity (uses config default if None)
            ego_param: Ego vehicle parameters (uses default if None)
            
        Returns:
            PlanningResult with trajectory and status
        """
        import time
        start_time = time.time()
        
        result = PlanningResult()
        target_vel = target_velocity or self.config.target_velocity
        ego_param = ego_param or VehicleParam()
        
        # Step 1: Set up state transformer with reference lane
        self.reference_lane = reference_lane
        self.state_transformer = StateTransformer(reference_lane)
        
        # Step 2: Transform initial state to Frenet frame
        try:
            self.initial_frenet_state = self.state_transformer.global_to_frenet(
                initial_state
            )
        except Exception as e:
            result.message = f"Failed to transform initial state: {e}"
            return result
        
        # Step 3: Reset and construct SSC map
        self.ssc_map.reset(self.initial_frenet_state)
        
        # Transform obstacle trajectories to Frenet frame and fill map
        self._fill_obstacles_to_map(obstacle_trajectories, ego_param)
        
        # Step 4: Generate initial trajectory (simple constant velocity)
        initial_traj_fs = generate_initial_trajectory_fs(
            initial_fs=self.initial_frenet_state,
            target_velocity=target_vel,
            duration=self.config.planning_horizon,
            dt=self.config.planning_dt
        )
        
        # Step 5: Generate corridor
        corridor = self.corridor_generator.generate_corridor(initial_traj_fs)
        
        if not corridor.is_valid or len(corridor.cubes) == 0:
            result.message = "Failed to generate valid corridor"
            result.corridor = corridor
            return result
        
        result.corridor = corridor
        
        # Step 6: Convert corridor to QP constraints
        qp_corridors = self.corridor_generator.corridor_to_constraints(corridor)
        
        if not qp_corridors:
            result.message = "No corridor constraints generated"
            return result
        
        # Step 7: Set up QP optimization
        # Start constraints: position, velocity, acceleration
        start_constraints = [
            np.array([self.initial_frenet_state.vec_s[0], 
                     self.initial_frenet_state.vec_dt[0]]),
            np.array([max(self.initial_frenet_state.vec_s[1], 0.1),
                     self.initial_frenet_state.vec_dt[1]]),
            np.array([self.initial_frenet_state.vec_s[2],
                     self.initial_frenet_state.vec_dt[2]])
        ]
        
        # End constraints: position, velocity (from last trajectory point)
        last_fs = initial_traj_fs[-1].frenet_state
        end_constraints = [
            np.array([last_fs.vec_s[0], last_fs.vec_dt[0]]),
            np.array([max(last_fs.vec_s[1], 0.1), last_fs.vec_dt[1]])
        ]
        
        # Reference points for proximity cost
        ref_stamps = [fs_v.frenet_state.time_stamp for fs_v in initial_traj_fs]
        ref_points = [np.array([fs_v.frenet_state.vec_s[0], 
                               fs_v.frenet_state.vec_dt[0]]) 
                     for fs_v in initial_traj_fs]
        
        # Step 8: Run QP optimization
        try:
            trajectory = optimize_bezier_in_corridor(
                start_constraints=start_constraints,
                end_constraints=end_constraints,
                corridors=qp_corridors,
                ref_stamps=ref_stamps,
                ref_points=ref_points,
                weight_proximity=self.config.weight_proximity,
                degree=self.config.bezier_degree
            )
        except Exception as e:
            result.message = f"QP optimization failed: {e}"
            return result
        
        if trajectory is None:
            result.message = "QP optimization returned no solution"
            return result
        
        result.trajectory = trajectory
        
        # Step 9: Sample trajectory and convert to Cartesian
        result.frenet_states = self._sample_trajectory(trajectory)
        result.cartesian_states = self._convert_to_cartesian(result.frenet_states)
        
        result.success = True
        result.message = "Planning successful"
        result.time_cost = (time.time() - start_time) * 1000  # ms
        
        return result
    
    def _fill_obstacles_to_map(self, 
                               obstacle_trajectories: List[ObstacleTrajectory],
                               ego_param: VehicleParam):
        """Fill obstacle trajectories into SSC map.
        
        Args:
            obstacle_trajectories: List of obstacle trajectories
            ego_param: Ego vehicle parameters (for inflation)
        """
        for obs_traj in obstacle_trajectories:
            # Sample obstacle trajectory at map time resolution
            t_start = self.ssc_map.start_time
            t_end = t_start + self.config.planning_horizon
            dt = self.config.map_config.map_resolution[2]
            
            fs_trajectory = []
            t = t_start
            
            while t <= t_end:
                # Get obstacle state at time t
                obs_state = obs_traj.get_state_at_time(t)
                if obs_state is None:
                    t += dt
                    continue
                
                # Transform to Frenet frame
                try:
                    obs_fs = self.state_transformer.global_to_frenet(obs_state)
                except:
                    t += dt
                    continue
                
                # Get vehicle vertices in Frenet frame
                vertices_global = obs_traj.param.get_vertices(obs_state)
                vertices_fs = self.state_transformer.transform_vehicle_vertices(
                    vertices_global, s_hint=obs_fs.vec_s[0]
                )
                
                fs_vehicle = FsVehicle(
                    frenet_state=obs_fs,
                    vertices=vertices_fs
                )
                fs_trajectory.append(fs_vehicle)
                
                t += dt
            
            # Fill trajectory into map
            self.ssc_map.fill_obstacle_trajectory_fs(fs_trajectory)
    
    def _sample_trajectory(self, trajectory: BezierSpline) -> List[FrenetState]:
        """Sample the optimized trajectory.
        
        Args:
            trajectory: Bezier spline trajectory
            
        Returns:
            List of sampled FrenetState
        """
        states = []
        
        t = trajectory.begin
        t_end = trajectory.end
        dt = self.config.planning_dt
        
        while t <= t_end:
            # Evaluate position, velocity, acceleration
            pos = trajectory.evaluate(t)  # (s, d)
            vel = trajectory.derivative(t, 1)  # (s_dot, d_dot)
            acc = trajectory.derivative(t, 2)  # (s_ddot, d_ddot)
            
            # Create FrenetState
            vec_s = np.array([pos[0], vel[0], acc[0]])
            vec_dt = np.array([pos[1], vel[1], acc[1]])
            
            fs = FrenetState.from_s_dt(vec_s, vec_dt, time_stamp=t)
            states.append(fs)
            
            t += dt
        
        return states
    
    def _convert_to_cartesian(self, 
                             frenet_states: List[FrenetState]) -> List[State]:
        """Convert Frenet states to Cartesian.
        
        Args:
            frenet_states: List of FrenetState
            
        Returns:
            List of Cartesian State
        """
        if self.state_transformer is None:
            return []
        
        cartesian_states = []
        for fs in frenet_states:
            try:
                state = self.state_transformer.frenet_to_global(fs)
                cartesian_states.append(state)
            except:
                continue
        
        return cartesian_states
    
    def get_trajectory_at_time(self, t: float) -> Optional[State]:
        """Get planned trajectory state at a specific time.
        
        Args:
            t: Time stamp
            
        Returns:
            State at time t, or None if not available
        """
        if (self.state_transformer is None or 
            not hasattr(self, '_last_result') or 
            self._last_result.trajectory is None):
            return None
        
        traj = self._last_result.trajectory
        
        if t < traj.begin or t > traj.end:
            return None
        
        pos = traj.evaluate(t)
        vel = traj.derivative(t, 1)
        acc = traj.derivative(t, 2)
        
        vec_s = np.array([pos[0], vel[0], acc[0]])
        vec_dt = np.array([pos[1], vel[1], acc[1]])
        
        fs = FrenetState.from_s_dt(vec_s, vec_dt, time_stamp=t)
        
        return self.state_transformer.frenet_to_global(fs)


def create_straight_lane(length: float = 200.0, 
                        num_points: int = 100) -> Lane:
    """Create a simple straight reference lane.
    
    Args:
        length: Lane length in meters
        num_points: Number of waypoints
        
    Returns:
        Straight Lane along x-axis
    """
    points = np.zeros((num_points, 2))
    points[:, 0] = np.linspace(0, length, num_points)
    return Lane.from_points(points)


def create_curved_lane(radius: float = 100.0,
                      angle_range: Tuple[float, float] = (0, np.pi/2),
                      num_points: int = 100) -> Lane:
    """Create a curved reference lane (circular arc).
    
    Args:
        radius: Arc radius in meters
        angle_range: (start_angle, end_angle) in radians
        num_points: Number of waypoints
        
    Returns:
        Curved Lane
    """
    angles = np.linspace(angle_range[0], angle_range[1], num_points)
    points = np.zeros((num_points, 2))
    points[:, 0] = radius * np.cos(angles)
    points[:, 1] = radius * np.sin(angles)
    return Lane.from_points(points)
