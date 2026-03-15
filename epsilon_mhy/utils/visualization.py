"""Visualization utilities for SSC planner.

Provides matplotlib-based visualization for:
- Trajectories in Cartesian and Frenet frames
- SSC map (spatio-temporal occupancy)
- Corridors
- Vehicles and obstacles
"""

from typing import List, Optional, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.collections import PatchCollection
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Visualization will not work.")

from ..core.state import State, FrenetState
from ..core.lane import Lane
from ..core.vehicle import Vehicle, VehicleParam, ObstacleTrajectory
from ..planning.ssc_map import SscMap
from ..planning.corridor import DrivingCorridor
from ..math.bezier import BezierSpline


class Visualizer:
    """Visualizer for SSC planner results."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """Initialize visualizer.
        
        Args:
            figsize: Figure size (width, height)
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for visualization")
        
        self.figsize = figsize
        self.colors = {
            'ego': 'blue',
            'obstacle': 'red',
            'lane': 'gray',
            'trajectory': 'green',
            'corridor': 'lightblue',
            'reference': 'orange'
        }
    
    def plot_scenario(self,
                     lane: Lane,
                     ego_state: State,
                     obstacles: List[ObstacleTrajectory],
                     trajectory_states: Optional[List[State]] = None,
                     title: str = "Planning Scenario") -> plt.Figure:
        """Plot the planning scenario in Cartesian coordinates.
        
        Args:
            lane: Reference lane
            ego_state: Current ego vehicle state
            obstacles: List of obstacle trajectories
            trajectory_states: Planned trajectory states
            title: Plot title
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot reference lane
        if lane.waypoints is not None and len(lane.waypoints) > 0:
            ax.plot(lane.waypoints[:, 0], lane.waypoints[:, 1],
                   '--', color=self.colors['lane'], linewidth=2, label='Reference Lane')
        
        # Plot ego vehicle
        self._plot_vehicle(ax, ego_state, VehicleParam(), 
                          color=self.colors['ego'], label='Ego')
        
        # Plot obstacles at initial time
        for i, obs in enumerate(obstacles):
            if obs.states:
                self._plot_vehicle(ax, obs.states[0], obs.param,
                                  color=self.colors['obstacle'],
                                  label=f'Obstacle {i}' if i == 0 else None)
        
        # Plot planned trajectory
        if trajectory_states:
            traj_x = [s.position[0] for s in trajectory_states]
            traj_y = [s.position[1] for s in trajectory_states]
            ax.plot(traj_x, traj_y, '-', color=self.colors['trajectory'],
                   linewidth=2, label='Planned Trajectory')
            
            # Plot trajectory endpoints
            ax.plot(traj_x[0], traj_y[0], 'go', markersize=10)
            ax.plot(traj_x[-1], traj_y[-1], 'g*', markersize=15)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(title)
        ax.legend()
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_frenet_trajectory(self,
                              frenet_states: List[FrenetState],
                              corridor_constraints: Optional[List] = None,
                              title: str = "Frenet Trajectory") -> plt.Figure:
        """Plot trajectory in Frenet coordinates (s-t and d-t).
        
        Args:
            frenet_states: List of FrenetState
            corridor_constraints: List of corridor constraints
            title: Plot title
            
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        times = [fs.time_stamp for fs in frenet_states]
        s_vals = [fs.vec_s[0] for fs in frenet_states]
        d_vals = [fs.vec_dt[0] for fs in frenet_states]
        v_vals = [fs.vec_s[1] for fs in frenet_states]
        a_vals = [fs.vec_s[2] for fs in frenet_states]
        
        # s-t plot
        ax1 = axes[0, 0]
        ax1.plot(times, s_vals, '-b', linewidth=2, label='s(t)')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('s (m)')
        ax1.set_title('Longitudinal Position')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # d-t plot
        ax2 = axes[0, 1]
        ax2.plot(times, d_vals, '-g', linewidth=2, label='d(t)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('d (m)')
        ax2.set_title('Lateral Deviation')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Velocity plot
        ax3 = axes[1, 0]
        ax3.plot(times, v_vals, '-r', linewidth=2, label='v(t)')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Velocity (m/s)')
        ax3.set_title('Longitudinal Velocity')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # s-d plot
        ax4 = axes[1, 1]
        ax4.plot(s_vals, d_vals, '-m', linewidth=2, label='Trajectory')
        ax4.plot(s_vals[0], d_vals[0], 'go', markersize=10, label='Start')
        ax4.plot(s_vals[-1], d_vals[-1], 'r*', markersize=15, label='End')
        ax4.set_xlabel('s (m)')
        ax4.set_ylabel('d (m)')
        ax4.set_title('s-d Trajectory')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        fig.suptitle(title)
        fig.tight_layout()
        
        return fig
    
    def plot_ssc_map_slice(self,
                          ssc_map: SscMap,
                          t_idx: int,
                          title: str = "SSC Map Slice") -> plt.Figure:
        """Plot a time slice of the SSC map.
        
        Args:
            ssc_map: SSC map
            t_idx: Time index to visualize
            title: Plot title
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        grid_slice = ssc_map.get_time_slice(t_idx)
        
        # Create extent for proper axis labels
        s_min = ssc_map.get_s_from_index(0)
        s_max = ssc_map.get_s_from_index(ssc_map.config.map_size[0])
        d_min = ssc_map.get_d_from_index(0)
        d_max = ssc_map.get_d_from_index(ssc_map.config.map_size[1])
        
        im = ax.imshow(grid_slice.T, origin='lower', aspect='auto',
                      extent=[s_min, s_max, d_min, d_max],
                      cmap='RdYlGn_r', vmin=0, vmax=100)
        
        t = ssc_map.get_time_from_index(t_idx)
        ax.set_xlabel('s (m)')
        ax.set_ylabel('d (m)')
        ax.set_title(f'{title} (t = {t:.2f}s)')
        
        plt.colorbar(im, ax=ax, label='Occupancy')
        
        return fig
    
    def plot_ssc_map_3d(self,
                       ssc_map: SscMap,
                       title: str = "3D SSC Map") -> plt.Figure:
        """Plot 3D visualization of SSC map.
        
        Args:
            ssc_map: SSC map
            title: Plot title
            
        Returns:
            matplotlib Figure
        """
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Get occupied cells
        occupied = np.where(ssc_map.grid == SscMap.OCCUPIED)
        
        if len(occupied[0]) > 0:
            # Convert indices to world coordinates
            s_vals = [ssc_map.get_s_from_index(i) for i in occupied[0]]
            d_vals = [ssc_map.get_d_from_index(i) for i in occupied[1]]
            t_vals = [ssc_map.get_time_from_index(i) for i in occupied[2]]
            
            # Subsample if too many points
            max_points = 5000
            if len(s_vals) > max_points:
                indices = np.random.choice(len(s_vals), max_points, replace=False)
                s_vals = [s_vals[i] for i in indices]
                d_vals = [d_vals[i] for i in indices]
                t_vals = [t_vals[i] for i in indices]
            
            ax.scatter(s_vals, d_vals, t_vals, c='red', alpha=0.3, s=1)
        
        ax.set_xlabel('s (m)')
        ax.set_ylabel('d (m)')
        ax.set_zlabel('t (s)')
        ax.set_title(title)
        
        return fig
    
    def plot_corridor(self,
                     corridor: DrivingCorridor,
                     ssc_map: SscMap,
                     trajectory_fs: Optional[List[FrenetState]] = None,
                     title: str = "Driving Corridor") -> plt.Figure:
        """Plot driving corridor with trajectory.
        
        Args:
            corridor: Driving corridor
            ssc_map: SSC map (for coordinate conversion)
            trajectory_fs: Trajectory in Frenet coordinates
            title: Plot title
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot corridor cubes (projected onto s-t plane)
        for cube_data in corridor.cubes:
            cube = cube_data.cube
            
            s_lb = ssc_map.get_s_from_index(cube.lower_bound[0])
            s_ub = ssc_map.get_s_from_index(cube.upper_bound[0])
            t_lb = ssc_map.get_time_from_index(cube.lower_bound[2])
            t_ub = ssc_map.get_time_from_index(cube.upper_bound[2])
            
            rect = patches.Rectangle(
                (t_lb, s_lb), t_ub - t_lb, s_ub - s_lb,
                linewidth=1, edgecolor='blue', facecolor=self.colors['corridor'],
                alpha=0.5
            )
            ax.add_patch(rect)
        
        # Plot trajectory
        if trajectory_fs:
            times = [fs.time_stamp for fs in trajectory_fs]
            s_vals = [fs.vec_s[0] for fs in trajectory_fs]
            ax.plot(times, s_vals, '-', color=self.colors['trajectory'],
                   linewidth=2, label='Trajectory')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('s (m)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_animation_frame(self,
                            ax: plt.Axes,
                            lane: Lane,
                            ego_state: State,
                            obstacles: List[ObstacleTrajectory],
                            time: float,
                            trajectory_states: Optional[List[State]] = None):
        """Plot a single frame for animation.
        
        Args:
            ax: Matplotlib axes
            lane: Reference lane
            ego_state: Current ego state
            obstacles: Obstacle trajectories
            time: Current time
            trajectory_states: Planned trajectory
        """
        ax.clear()
        
        # Plot lane
        if lane.waypoints is not None and len(lane.waypoints) > 0:
            ax.plot(lane.waypoints[:, 0], lane.waypoints[:, 1],
                   '--', color=self.colors['lane'], linewidth=2)
        
        # Plot ego
        self._plot_vehicle(ax, ego_state, VehicleParam(), 
                          color=self.colors['ego'])
        
        # Plot obstacles at current time
        for obs in obstacles:
            obs_state = obs.get_state_at_time(time)
            if obs_state:
                self._plot_vehicle(ax, obs_state, obs.param,
                                  color=self.colors['obstacle'])
        
        # Plot trajectory
        if trajectory_states:
            traj_x = [s.position[0] for s in trajectory_states]
            traj_y = [s.position[1] for s in trajectory_states]
            ax.plot(traj_x, traj_y, '-', color=self.colors['trajectory'],
                   linewidth=2, alpha=0.5)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Time: {time:.2f}s')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
    
    def _plot_vehicle(self, ax: plt.Axes, state: State, 
                     param: VehicleParam, color: str = 'blue',
                     label: Optional[str] = None):
        """Plot a vehicle as a rectangle.
        
        Args:
            ax: Matplotlib axes
            state: Vehicle state
            param: Vehicle parameters
            color: Fill color
            label: Optional label
        """
        vertices = param.get_vertices(state)
        
        polygon = patches.Polygon(vertices, closed=True,
                                  facecolor=color, edgecolor='black',
                                  alpha=0.7, label=label)
        ax.add_patch(polygon)
        
        # Draw heading direction
        x, y = state.position
        dx = 0.5 * param.length * np.cos(state.angle)
        dy = 0.5 * param.length * np.sin(state.angle)
        ax.arrow(x, y, dx, dy, head_width=0.3, head_length=0.2,
                fc=color, ec='black')


def quick_plot_result(result: "PlanningResult",
                     lane: Lane,
                     initial_state: State,
                     obstacles: List[ObstacleTrajectory]) -> List[plt.Figure]:
    """Quick visualization of planning result.
    
    Args:
        result: Planning result
        lane: Reference lane
        initial_state: Initial ego state
        obstacles: Obstacle trajectories
        
    Returns:
        List of figures
    """
    from ..planning.ssc_planner import PlanningResult
    
    vis = Visualizer()
    figures = []
    
    # Plot scenario
    fig1 = vis.plot_scenario(
        lane=lane,
        ego_state=initial_state,
        obstacles=obstacles,
        trajectory_states=result.cartesian_states if result.success else None,
        title=f"Planning Result: {result.message}"
    )
    figures.append(fig1)
    
    # Plot Frenet trajectory if successful
    if result.success and result.frenet_states:
        fig2 = vis.plot_frenet_trajectory(
            frenet_states=result.frenet_states,
            title="Planned Trajectory in Frenet Frame"
        )
        figures.append(fig2)
    
    return figures
