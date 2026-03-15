"""Corridor generation and inflation algorithms.

Reference: EPSILON/util/ssc_planner/src/ssc_planner/ssc_map.cc

The corridor generation algorithm:
1. Takes seed points from an initial trajectory
2. Creates initial cubes containing consecutive seed points
3. Inflates cubes in free space while maintaining safety
4. Returns a sequence of corridors for trajectory optimization
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

from .ssc_map import SscMap, SscMapConfig, AxisAlignedCube
from ..core.state import FrenetState
from ..core.vehicle import FsVehicle
from ..math.qp_solver import Corridor


@dataclass
class DrivingCube:
    """A cube in the driving corridor with seed points.
    
    Attributes:
        cube: Axis-aligned cube in grid coordinates
        seeds: List of seed points (trajectory waypoints) in grid coordinates
    """
    cube: AxisAlignedCube
    seeds: List[np.ndarray] = field(default_factory=list)


@dataclass
class DrivingCorridor:
    """A sequence of cubes forming a driving corridor.
    
    Attributes:
        cubes: List of driving cubes
        is_valid: Whether the corridor is valid (no collisions)
    """
    cubes: List[DrivingCube] = field(default_factory=list)
    is_valid: bool = True


class CorridorGenerator:
    """Generates safe corridors from SSC map and initial trajectory.
    
    The algorithm:
    1. Sample seed points along the initial trajectory
    2. Create initial cubes between consecutive seeds
    3. Inflate cubes in the free space
    4. Connect cubes to form continuous corridor
    """
    
    def __init__(self, ssc_map: SscMap, config: Optional[SscMapConfig] = None):
        """Initialize corridor generator.
        
        Args:
            ssc_map: The SSC map containing obstacles
            config: Map configuration (if None, use ssc_map's config)
        """
        self.ssc_map = ssc_map
        self.config = config or ssc_map.config
    
    def generate_corridor(self, 
                         initial_trajectory_fs: List[FsVehicle]) -> DrivingCorridor:
        """Generate driving corridor from initial trajectory.
        
        Args:
            initial_trajectory_fs: Initial trajectory in Frenet coordinates
            
        Returns:
            DrivingCorridor with inflated cubes
        """
        # Stage 1: Get seed points in grid coordinates
        seeds = self._get_trajectory_seeds(initial_trajectory_fs)
        
        if len(seeds) < 2:
            return DrivingCorridor(is_valid=False)
        
        # Stage 2: Generate and inflate cubes
        corridor = DrivingCorridor()
        
        i = 0
        while i < len(seeds) - 1:
            # Create initial cube between consecutive seeds
            seed_0 = seeds[i]
            seed_1 = seeds[i + 1]
            
            cube = self._create_initial_cube(seed_0, seed_1)
            
            # Check if initial cube is free
            if not self.ssc_map.is_cube_free(cube):
                corridor.is_valid = False
                driving_cube = DrivingCube(cube=cube, seeds=[seed_0, seed_1])
                corridor.cubes.append(driving_cube)
                break
            
            # Inflate the cube
            inflated_cube = self._inflate_cube(cube)
            
            # Create driving cube
            driving_cube = DrivingCube(cube=inflated_cube, seeds=[seed_0])
            
            # Try to include more seeds in this cube
            j = i + 1
            while j < len(seeds):
                if self._cube_contains_point(inflated_cube, seeds[j]):
                    driving_cube.seeds.append(seeds[j])
                    j += 1
                else:
                    break
            
            # Cut cube at last seed's time
            if len(driving_cube.seeds) > 0:
                last_seed = driving_cube.seeds[-1]
                inflated_cube.upper_bound[2] = last_seed[2]
            
            corridor.cubes.append(driving_cube)
            i = j - 1 if j > i + 1 else i + 1
        
        if corridor.cubes and corridor.is_valid:
            # Ensure last cube ends at last seed
            last_seed = seeds[-1]
            corridor.cubes[-1].cube.upper_bound[2] = last_seed[2]
        
        return corridor
    
    def _get_trajectory_seeds(self, 
                             trajectory_fs: List[FsVehicle]) -> List[np.ndarray]:
        """Extract seed points from trajectory.
        
        Args:
            trajectory_fs: Trajectory in Frenet coordinates
            
        Returns:
            List of seed points in grid coordinates
        """
        seeds = []
        first_valid_found = False
        
        # Add initial state as first seed
        if self.ssc_map.initial_fs is not None:
            fs = self.ssc_map.initial_fs
            s = fs.vec_s[0]
            d = fs.vec_dt[0]
            t = fs.time_stamp
            world_pt = np.array([s, d, t])
            coord = self.ssc_map._world_to_grid(world_pt)
            
            if self.ssc_map._coord_in_range(coord):
                seeds.append(coord)
                first_valid_found = True
        
        for fs_vehicle in trajectory_fs:
            s = fs_vehicle.frenet_state.vec_s[0]
            d = fs_vehicle.frenet_state.vec_dt[0]
            t = fs_vehicle.frenet_state.time_stamp
            
            world_pt = np.array([s, d, t])
            coord = self.ssc_map._world_to_grid(world_pt)
            
            if not self.ssc_map._coord_in_range(coord):
                continue
            
            # Skip if time index <= 0 and we haven't found first valid seed
            if coord[2] <= 0 and not first_valid_found:
                continue
            
            first_valid_found = True
            seeds.append(coord)
        
        return seeds
    
    def _create_initial_cube(self, seed_0: np.ndarray, 
                            seed_1: np.ndarray) -> AxisAlignedCube:
        """Create initial cube containing two consecutive seeds.
        
        Args:
            seed_0: First seed point in grid coordinates
            seed_1: Second seed point in grid coordinates
            
        Returns:
            Axis-aligned cube
        """
        lower_bound = np.minimum(seed_0, seed_1)
        upper_bound = np.maximum(seed_0, seed_1)
        
        return AxisAlignedCube(lower_bound=lower_bound, upper_bound=upper_bound)
    
    def _inflate_cube(self, cube: AxisAlignedCube) -> AxisAlignedCube:
        """Inflate cube in free space.
        
        Args:
            cube: Initial cube
            
        Returns:
            Inflated cube
        """
        result = AxisAlignedCube(
            lower_bound=cube.lower_bound.copy(),
            upper_bound=cube.upper_bound.copy()
        )
        
        inflate_steps = self.config.inflate_steps
        max_grids_time = self.config.max_grids_along_time
        
        # Direction: [s+, s-, d+, d-, t+, t-]
        s_pos_done = False
        s_neg_done = False
        d_pos_done = False
        d_neg_done = False
        t_pos_done = False
        
        # Compute velocity-based s bounds
        t_idx = result.lower_bound[2] + max_grids_time
        dt = t_idx * self.config.map_resolution[2]
        
        if self.ssc_map.initial_fs is not None:
            s_init = self.ssc_map.initial_fs.vec_s[0]
            v_init = self.ssc_map.initial_fs.vec_s[1]
            a_max = self.config.max_lon_acc
            a_min = self.config.max_lon_dec
            
            s_upper = s_init + v_init * dt + 0.5 * a_max * dt * dt + v_init
            s_lower = s_init + v_init * dt + 0.5 * a_min * dt * dt - v_init
            
            s_idx_upper = self.ssc_map._world_to_grid_single(s_upper, 0)
            s_idx_lower = max(
                self.ssc_map._world_to_grid_single(s_lower, 0),
                int(self.config.s_back_len / 2.0 / self.config.map_resolution[0])
            )
        else:
            s_idx_upper = self.config.map_size[0] - 1
            s_idx_lower = 0
        
        # Iterative inflation
        while not (s_pos_done and s_neg_done and d_pos_done and d_neg_done):
            # Inflate in s+ direction
            if not s_pos_done:
                s_pos_done = self._inflate_direction(result, 0, 1, inflate_steps[0])
                if result.upper_bound[0] >= s_idx_upper:
                    s_pos_done = True
            
            # Inflate in s- direction
            if not s_neg_done:
                s_neg_done = self._inflate_direction(result, 0, -1, inflate_steps[1])
                if result.lower_bound[0] <= s_idx_lower:
                    s_neg_done = True
            
            # Inflate in d+ direction
            if not d_pos_done:
                d_pos_done = self._inflate_direction(result, 1, 1, inflate_steps[2])
            
            # Inflate in d- direction
            if not d_neg_done:
                d_neg_done = self._inflate_direction(result, 1, -1, inflate_steps[3])
        
        # Inflate in t+ direction
        while not t_pos_done:
            t_pos_done = self._inflate_direction(result, 2, 1, inflate_steps[4])
            
            # Check time limit
            if (result.upper_bound[2] - result.lower_bound[2] >= max_grids_time):
                t_pos_done = True
        
        return result
    
    def _inflate_direction(self, cube: AxisAlignedCube, 
                          axis: int, direction: int,
                          n_steps: int) -> bool:
        """Inflate cube in one direction.
        
        Args:
            cube: Cube to inflate (modified in place)
            axis: Axis to inflate (0=s, 1=d, 2=t)
            direction: +1 for positive direction, -1 for negative
            n_steps: Number of steps to try
            
        Returns:
            True if inflation stopped (hit obstacle or boundary)
        """
        map_size = self.config.map_size
        
        for _ in range(n_steps):
            if direction > 0:
                new_idx = cube.upper_bound[axis] + 1
                if new_idx >= map_size[axis]:
                    return True
                
                # Check if new plane is free
                if self._is_plane_free(cube, axis, new_idx):
                    cube.upper_bound[axis] = new_idx
                else:
                    return True
            else:
                new_idx = cube.lower_bound[axis] - 1
                if new_idx < 0:
                    return True
                
                if self._is_plane_free(cube, axis, new_idx):
                    cube.lower_bound[axis] = new_idx
                else:
                    return True
        
        return False
    
    def _is_plane_free(self, cube: AxisAlignedCube, 
                      axis: int, idx: int) -> bool:
        """Check if a plane perpendicular to axis at idx is free.
        
        Args:
            cube: Current cube
            axis: Perpendicular axis
            idx: Index along the axis
            
        Returns:
            True if all cells in the plane are free
        """
        # Get other two axes
        other_axes = [i for i in range(3) if i != axis]
        ax1, ax2 = other_axes
        
        for i in range(cube.lower_bound[ax1], cube.upper_bound[ax1] + 1):
            for j in range(cube.lower_bound[ax2], cube.upper_bound[ax2] + 1):
                coord = np.zeros(3, dtype=np.int32)
                coord[axis] = idx
                coord[ax1] = i
                coord[ax2] = j
                
                if not self.ssc_map.is_free(coord):
                    return False
        
        return True
    
    def _cube_contains_point(self, cube: AxisAlignedCube, 
                            point: np.ndarray) -> bool:
        """Check if cube contains a point.
        
        Args:
            cube: Axis-aligned cube
            point: Point in grid coordinates
            
        Returns:
            True if point is inside or on boundary of cube
        """
        return cube.contains(point)
    
    def corridor_to_constraints(self, 
                               corridor: DrivingCorridor) -> List[Corridor]:
        """Convert driving corridor to QP constraints.
        
        Args:
            corridor: Driving corridor in grid coordinates
            
        Returns:
            List of Corridor objects for QP optimization
        """
        constraints = []
        
        for driving_cube in corridor.cubes:
            cube = driving_cube.cube
            
            # Convert grid bounds to world coordinates
            t_lb = self.ssc_map._grid_to_world_single(cube.lower_bound[2], 2)
            t_ub = self.ssc_map._grid_to_world_single(cube.upper_bound[2], 2)
            
            s_lb = self.ssc_map._grid_to_world_single(cube.lower_bound[0], 0)
            s_ub = self.ssc_map._grid_to_world_single(cube.upper_bound[0], 0)
            
            d_lb = self.ssc_map._grid_to_world_single(cube.lower_bound[1], 1)
            d_ub = self.ssc_map._grid_to_world_single(cube.upper_bound[1], 1)
            
            p_lb = np.array([s_lb, d_lb])
            p_ub = np.array([s_ub, d_ub])
            
            # Velocity and acceleration bounds from config
            v_lb = np.array([self.config.min_lon_vel, -self.config.max_lat_vel])
            v_ub = np.array([self.config.max_lon_vel, self.config.max_lat_vel])
            a_lb = np.array([self.config.max_lon_dec, -self.config.max_lat_acc])
            a_ub = np.array([self.config.max_lon_acc, self.config.max_lat_acc])
            
            constraints.append(Corridor(
                t_lb=t_lb,
                t_ub=t_ub,
                p_lb=p_lb,
                p_ub=p_ub,
                v_lb=v_lb,
                v_ub=v_ub,
                a_lb=a_lb,
                a_ub=a_ub
            ))
        
        return constraints


def generate_initial_trajectory_fs(
    initial_fs: FrenetState,
    target_velocity: float,
    duration: float,
    dt: float = 0.1
) -> List[FsVehicle]:
    """Generate a simple initial trajectory in Frenet coordinates.
    
    Creates a constant-velocity trajectory with lateral position
    converging to zero.
    
    Args:
        initial_fs: Initial Frenet state
        target_velocity: Target longitudinal velocity
        duration: Total duration
        dt: Time step
        
    Returns:
        List of FsVehicle representing the trajectory
    """
    trajectory = []
    
    s = initial_fs.vec_s[0]
    d = initial_fs.vec_dt[0]
    v = initial_fs.vec_s[1]
    t = initial_fs.time_stamp
    
    # Simple velocity transition
    n_steps = int(duration / dt)
    
    for i in range(n_steps + 1):
        # Create Frenet state
        fs = FrenetState.from_s_dt(
            s=np.array([s, v, 0.0]),
            dt=np.array([d, -d * 0.5, 0.0]),  # Converge to center
            time_stamp=t
        )
        
        # Create FsVehicle (simplified - no actual vertices)
        fs_vehicle = FsVehicle(
            frenet_state=fs,
            vertices=np.array([
                [s - 2, d - 1],
                [s + 2, d - 1],
                [s + 2, d + 1],
                [s - 2, d + 1]
            ])
        )
        
        trajectory.append(fs_vehicle)
        
        # Update state
        t += dt
        s += v * dt
        d *= 0.95  # Exponential convergence to center
        
        # Velocity transition
        v = v + (target_velocity - v) * 0.1
    
    return trajectory
