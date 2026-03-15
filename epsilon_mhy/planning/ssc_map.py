"""Spatio-temporal Semantic Corridor Map.

Reference: EPSILON/util/ssc_planner/src/ssc_planner/ssc_map.cc

The SSC map is a 3D grid in Frenet-time space (s, d, t).
It is used to:
1. Represent obstacles in spatio-temporal space
2. Generate safe corridors for trajectory optimization
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np

from ..core.state import FrenetState
from ..core.vehicle import FsVehicle, ObstacleTrajectory, VehicleParam
from ..core.types import ErrorType


@dataclass
class SscMapConfig:
    """Configuration for SSC Map.
    
    Attributes:
        map_size: Grid size (s, d, t) in cells
        map_resolution: Cell size (s, d, t) in meters/seconds
        s_back_len: Distance behind ego to include
        max_lon_vel: Maximum longitudinal velocity
        min_lon_vel: Minimum longitudinal velocity
        max_lon_acc: Maximum longitudinal acceleration
        max_lon_dec: Maximum longitudinal deceleration (negative)
        max_lat_vel: Maximum lateral velocity
        max_lat_acc: Maximum lateral acceleration
        inflate_steps: Inflation steps in each direction [s+, s-, d+, d-, t+, t-]
        max_grids_along_time: Maximum time grids per corridor cube
    """
    map_size: Tuple[int, int, int] = (400, 50, 81)  # s, d, t
    map_resolution: Tuple[float, float, float] = (0.25, 0.2, 0.1)  # m, m, s
    s_back_len: float = 5.0
    max_lon_vel: float = 50.0
    min_lon_vel: float = 0.1
    max_lon_acc: float = 3.0
    max_lon_dec: float = -8.0
    max_lat_vel: float = 3.0
    max_lat_acc: float = 2.5
    inflate_steps: Tuple[int, ...] = (20, 5, 10, 10, 1, 1)
    max_grids_along_time: int = 2
    
    @property
    def s_range(self) -> float:
        """Total s range covered by the map."""
        return self.map_size[0] * self.map_resolution[0]
    
    @property
    def d_range(self) -> float:
        """Total d range covered by the map."""
        return self.map_size[1] * self.map_resolution[1]
    
    @property
    def t_range(self) -> float:
        """Total time range covered by the map."""
        return self.map_size[2] * self.map_resolution[2]


@dataclass
class AxisAlignedCube:
    """Axis-aligned cube in 3D grid space.
    
    Represents a rectangular region in (s, d, t) grid coordinates.
    """
    lower_bound: np.ndarray  # (3,) int - (s_min, d_min, t_min)
    upper_bound: np.ndarray  # (3,) int - (s_max, d_max, t_max)
    
    def __post_init__(self):
        self.lower_bound = np.asarray(self.lower_bound, dtype=np.int32)
        self.upper_bound = np.asarray(self.upper_bound, dtype=np.int32)
    
    def contains(self, point: np.ndarray) -> bool:
        """Check if a point is inside the cube."""
        return np.all(point >= self.lower_bound) and np.all(point <= self.upper_bound)
    
    @property
    def volume(self) -> int:
        """Number of cells in the cube."""
        return int(np.prod(self.upper_bound - self.lower_bound + 1))


class SscMap:
    """3D Spatio-temporal Semantic Corridor Map.
    
    The map is a 3D grid where:
    - Axis 0 (s): longitudinal position in Frenet frame
    - Axis 1 (d): lateral position in Frenet frame
    - Axis 2 (t): time
    
    Grid values:
    - 0: Free
    - 100: Occupied by obstacle
    """
    
    OCCUPIED = 100
    FREE = 0
    
    def __init__(self, config: SscMapConfig):
        """Initialize SSC map.
        
        Args:
            config: Map configuration
        """
        self.config = config
        
        # 3D grid: (s, d, t)
        self.grid = np.zeros(config.map_size, dtype=np.uint8)
        
        # Map origin in world coordinates
        self.origin = np.zeros(3)  # (s_origin, d_origin, t_origin)
        
        # Initial state (updated when map is reset)
        self.initial_fs: Optional[FrenetState] = None
        self.start_time: float = 0.0
    
    def reset(self, initial_fs: FrenetState):
        """Reset map with new initial state.
        
        Args:
            initial_fs: Initial Frenet state of ego vehicle
        """
        self.grid.fill(self.FREE)
        self.initial_fs = initial_fs
        self.start_time = initial_fs.time_stamp
        
        # Set map origin
        self.origin[0] = initial_fs.vec_s[0] - self.config.s_back_len  # s origin
        self.origin[1] = -self.config.d_range / 2.0  # d origin (centered)
        self.origin[2] = initial_fs.time_stamp  # t origin
    
    def construct_map(self,
                     obstacle_trajectories: List[ObstacleTrajectory],
                     static_obstacles_fs: Optional[List[np.ndarray]] = None):
        """Construct the SSC map by filling in obstacles.
        
        Args:
            obstacle_trajectories: List of dynamic obstacle trajectories
            static_obstacles_fs: List of static obstacle points in Frenet [(s, d), ...]
        """
        self.grid.fill(self.FREE)
        
        # Fill static obstacles (if any)
        if static_obstacles_fs:
            self._fill_static_part(static_obstacles_fs)
        
        # Fill dynamic obstacles
        self._fill_dynamic_part(obstacle_trajectories)
    
    def _fill_static_part(self, obstacle_points_fs: List[np.ndarray]):
        """Fill static obstacles into the map.
        
        Static obstacles are extruded through all time slices.
        
        Args:
            obstacle_points_fs: List of (s, d) points in Frenet frame
        """
        for pt in obstacle_points_fs:
            s, d = pt[0], pt[1]
            if s <= 0:
                continue
            
            # Fill through all time
            for t_idx in range(self.config.map_size[2]):
                world_pt = np.array([s, d, t_idx * self.config.map_resolution[2]])
                coord = self._world_to_grid(world_pt)
                if self._coord_in_range(coord):
                    self.grid[coord[0], coord[1], coord[2]] = self.OCCUPIED
    
    def _fill_dynamic_part(self, obstacle_trajectories: List[ObstacleTrajectory]):
        """Fill dynamic obstacles into the map.
        
        Args:
            obstacle_trajectories: List of obstacle trajectories
        """
        for traj in obstacle_trajectories:
            self._fill_obstacle_trajectory(traj)
    
    def _fill_obstacle_trajectory(self, traj: ObstacleTrajectory):
        """Fill a single obstacle trajectory into the map.
        
        Args:
            traj: Obstacle trajectory
        """
        if not traj.states:
            return
        
        for state in traj.states:
            # Get vehicle vertices
            vertices = traj.param.get_vertices(state)
            
            # Time coordinate
            t_world = state.time_stamp
            t_grid = self._world_to_grid_single(t_world, 2)
            
            if not (0 <= t_grid < self.config.map_size[2]):
                continue
            
            # For simplicity, we'll use the bounding box of vertices
            # A more accurate approach would fill the polygon
            s_min = float('inf')
            s_max = float('-inf')
            d_min = float('inf')
            d_max = float('-inf')
            
            for v in vertices:
                # Note: vertices are in global frame, we need to convert to Frenet
                # For now, assume obstacle states are already compatible
                s_min = min(s_min, v[0])
                s_max = max(s_max, v[0])
                d_min = min(d_min, v[1])
                d_max = max(d_max, v[1])
            
            # Convert to grid coordinates
            s_idx_min = self._world_to_grid_single(s_min, 0)
            s_idx_max = self._world_to_grid_single(s_max, 0)
            d_idx_min = self._world_to_grid_single(d_min, 1)
            d_idx_max = self._world_to_grid_single(d_max, 1)
            
            # Fill grid cells
            for s_idx in range(max(0, s_idx_min), min(self.config.map_size[0], s_idx_max + 1)):
                for d_idx in range(max(0, d_idx_min), min(self.config.map_size[1], d_idx_max + 1)):
                    self.grid[s_idx, d_idx, t_grid] = self.OCCUPIED
    
    def fill_obstacle_trajectory_fs(self, 
                                    fs_trajectory: List[FsVehicle]):
        """Fill an obstacle trajectory already in Frenet coordinates.
        
        Args:
            fs_trajectory: List of FsVehicle (vehicle in Frenet frame)
        """
        for fs_vehicle in fs_trajectory:
            t_world = fs_vehicle.frenet_state.time_stamp
            t_grid = self._world_to_grid_single(t_world, 2)
            
            if not (0 <= t_grid < self.config.map_size[2]):
                continue
            
            # Get bounding box of vertices in Frenet frame
            if len(fs_vehicle.vertices) == 0:
                continue
            
            s_min = np.min(fs_vehicle.vertices[:, 0])
            s_max = np.max(fs_vehicle.vertices[:, 0])
            d_min = np.min(fs_vehicle.vertices[:, 1])
            d_max = np.max(fs_vehicle.vertices[:, 1])
            
            # Convert to grid coordinates
            s_idx_min = self._world_to_grid_single(s_min, 0)
            s_idx_max = self._world_to_grid_single(s_max, 0)
            d_idx_min = self._world_to_grid_single(d_min, 1)
            d_idx_max = self._world_to_grid_single(d_max, 1)
            
            # Fill grid cells
            for s_idx in range(max(0, s_idx_min), min(self.config.map_size[0], s_idx_max + 1)):
                for d_idx in range(max(0, d_idx_min), min(self.config.map_size[1], d_idx_max + 1)):
                    self.grid[s_idx, d_idx, t_grid] = self.OCCUPIED
    
    def is_free(self, coord: np.ndarray) -> bool:
        """Check if a grid cell is free.
        
        Args:
            coord: Grid coordinate (s_idx, d_idx, t_idx)
            
        Returns:
            True if free, False if occupied or out of range
        """
        if not self._coord_in_range(coord):
            return False
        return self.grid[coord[0], coord[1], coord[2]] == self.FREE
    
    def is_cube_free(self, cube: AxisAlignedCube) -> bool:
        """Check if all cells in a cube are free.
        
        Args:
            cube: Axis-aligned cube in grid coordinates
            
        Returns:
            True if all cells are free
        """
        lb = np.maximum(cube.lower_bound, 0)
        ub = np.minimum(cube.upper_bound, np.array(self.config.map_size) - 1)
        
        region = self.grid[lb[0]:ub[0]+1, lb[1]:ub[1]+1, lb[2]:ub[2]+1]
        return np.all(region == self.FREE)
    
    def _world_to_grid(self, world_pt: np.ndarray) -> np.ndarray:
        """Convert world (s, d, t) to grid coordinates.
        
        Args:
            world_pt: Point in world coordinates (s, d, t)
            
        Returns:
            Grid coordinate (s_idx, d_idx, t_idx)
        """
        grid_pt = np.zeros(3, dtype=np.int32)
        for i in range(3):
            grid_pt[i] = self._world_to_grid_single(world_pt[i], i)
        return grid_pt
    
    def _world_to_grid_single(self, value: float, dim: int) -> int:
        """Convert single world coordinate to grid index.
        
        Args:
            value: World coordinate value
            dim: Dimension (0=s, 1=d, 2=t)
            
        Returns:
            Grid index
        """
        return int((value - self.origin[dim]) / self.config.map_resolution[dim])
    
    def _grid_to_world(self, grid_pt: np.ndarray) -> np.ndarray:
        """Convert grid coordinates to world (s, d, t).
        
        Args:
            grid_pt: Grid coordinate
            
        Returns:
            World coordinate (s, d, t)
        """
        world_pt = np.zeros(3)
        for i in range(3):
            world_pt[i] = self._grid_to_world_single(grid_pt[i], i)
        return world_pt
    
    def _grid_to_world_single(self, idx: int, dim: int) -> float:
        """Convert single grid index to world coordinate.
        
        Args:
            idx: Grid index
            dim: Dimension
            
        Returns:
            World coordinate value
        """
        return self.origin[dim] + idx * self.config.map_resolution[dim]
    
    def _coord_in_range(self, coord: np.ndarray) -> bool:
        """Check if grid coordinate is within map bounds."""
        return (0 <= coord[0] < self.config.map_size[0] and
                0 <= coord[1] < self.config.map_size[1] and
                0 <= coord[2] < self.config.map_size[2])
    
    def get_time_slice(self, t_idx: int) -> np.ndarray:
        """Get a 2D slice of the map at a given time index.
        
        Args:
            t_idx: Time index
            
        Returns:
            2D array of shape (s_size, d_size)
        """
        if not (0 <= t_idx < self.config.map_size[2]):
            return np.zeros((self.config.map_size[0], self.config.map_size[1]))
        return self.grid[:, :, t_idx]
    
    def get_time_from_index(self, t_idx: int) -> float:
        """Convert time index to actual time."""
        return self.origin[2] + t_idx * self.config.map_resolution[2]
    
    def get_s_from_index(self, s_idx: int) -> float:
        """Convert s index to actual s value."""
        return self.origin[0] + s_idx * self.config.map_resolution[0]
    
    def get_d_from_index(self, d_idx: int) -> float:
        """Convert d index to actual d value."""
        return self.origin[1] + d_idx * self.config.map_resolution[1]
