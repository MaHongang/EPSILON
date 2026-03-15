"""Vehicle representation.

Reference: EPSILON/core/common/inc/common/basics/semantics.h
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

from .state import State, FrenetState
from .types import INVALID_AGENT_ID


@dataclass
class VehicleParam:
    """Vehicle geometric and dynamic parameters.
    
    Attributes:
        width: Vehicle width in meters
        length: Vehicle length in meters
        wheel_base: Distance between front and rear axles in meters
        front_suspension: Distance from front axle to front bumper
        rear_suspension: Distance from rear axle to rear bumper
        max_steering_angle: Maximum steering angle in degrees
        max_longitudinal_acc: Maximum longitudinal acceleration in m/s^2
        max_lateral_acc: Maximum lateral acceleration in m/s^2
        d_cr: Distance from geometry center to rear axle
    """
    width: float = 1.90
    length: float = 4.88
    wheel_base: float = 2.85
    front_suspension: float = 0.93
    rear_suspension: float = 1.10
    max_steering_angle: float = 45.0
    max_longitudinal_acc: float = 2.0
    max_lateral_acc: float = 2.0
    d_cr: float = 1.34
    
    def get_vertices(self, state: State) -> np.ndarray:
        """Get the four corners of the vehicle bounding box.
        
        Args:
            state: Current vehicle state (position at rear axle center)
            
        Returns:
            Array of shape (4, 2) with corners in order:
            [front_left, front_right, rear_right, rear_left]
        """
        x, y = state.position
        theta = state.angle
        
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        # Vehicle dimensions relative to rear axle center
        front_dist = self.length - self.rear_suspension
        rear_dist = self.rear_suspension
        half_width = self.width / 2
        
        # Corner offsets in vehicle frame
        corners_local = np.array([
            [front_dist, half_width],   # front left
            [front_dist, -half_width],  # front right
            [-rear_dist, -half_width],  # rear right
            [-rear_dist, half_width],   # rear left
        ])
        
        # Rotation matrix
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        
        # Transform to global frame
        corners_global = corners_local @ R.T + np.array([x, y])
        
        return corners_global
    
    def get_geometry_center(self, state: State) -> np.ndarray:
        """Get the geometry center position.
        
        Args:
            state: Current vehicle state (position at rear axle center)
            
        Returns:
            Position of geometry center (x, y)
        """
        dx = self.d_cr * np.cos(state.angle)
        dy = self.d_cr * np.sin(state.angle)
        return state.position + np.array([dx, dy])


@dataclass 
class Vehicle:
    """Vehicle with state and parameters.
    
    Attributes:
        id: Unique vehicle identifier
        param: Vehicle parameters
        state: Current vehicle state
    """
    id: int = INVALID_AGENT_ID
    param: VehicleParam = field(default_factory=VehicleParam)
    state: State = field(default_factory=State)
    
    def get_vertices(self) -> np.ndarray:
        """Get vehicle bounding box vertices."""
        return self.param.get_vertices(self.state)
    
    def get_geometry_center(self) -> np.ndarray:
        """Get vehicle geometry center."""
        return self.param.get_geometry_center(self.state)
    
    def copy(self) -> "Vehicle":
        """Create a deep copy of this vehicle."""
        return Vehicle(
            id=self.id,
            param=VehicleParam(**self.param.__dict__),
            state=self.state.copy()
        )


@dataclass
class FsVehicle:
    """Vehicle in Frenet coordinates.
    
    Attributes:
        frenet_state: State in Frenet coordinates
        vertices: Vehicle corners in Frenet frame [(s, d), ...]
    """
    frenet_state: FrenetState = field(default_factory=FrenetState)
    vertices: np.ndarray = field(default_factory=lambda: np.zeros((4, 2)))
    
    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=np.float64)


@dataclass
class ObstacleTrajectory:
    """Trajectory of a dynamic obstacle.
    
    Supports multiple input formats:
    - List of states with timestamps
    - List of (time, position, angle) tuples
    - List of (time, polygon) tuples for non-vehicle obstacles
    """
    id: int = INVALID_AGENT_ID
    states: List[State] = field(default_factory=list)
    param: VehicleParam = field(default_factory=VehicleParam)
    
    @classmethod
    def from_positions(cls, positions: List[Tuple[float, float, float, float]],
                      vehicle_param: Optional[VehicleParam] = None,
                      id: int = 0) -> "ObstacleTrajectory":
        """Create trajectory from list of (t, x, y, theta) tuples."""
        states = []
        for t, x, y, theta in positions:
            states.append(State(
                time_stamp=t,
                position=np.array([x, y]),
                angle=theta
            ))
        return cls(
            id=id,
            states=states,
            param=vehicle_param or VehicleParam()
        )
    
    @classmethod
    def from_state_list(cls, states: List[State],
                       vehicle_param: Optional[VehicleParam] = None,
                       id: int = 0) -> "ObstacleTrajectory":
        """Create trajectory from list of States."""
        return cls(
            id=id,
            states=states,
            param=vehicle_param or VehicleParam()
        )
    
    def get_state_at_time(self, t: float) -> Optional[State]:
        """Get interpolated state at given time.
        
        Returns None if t is outside the trajectory time range.
        """
        if not self.states:
            return None
        
        if t <= self.states[0].time_stamp:
            return self.states[0].copy()
        if t >= self.states[-1].time_stamp:
            return self.states[-1].copy()
        
        # Find bracketing states
        for i in range(len(self.states) - 1):
            t0 = self.states[i].time_stamp
            t1 = self.states[i + 1].time_stamp
            if t0 <= t <= t1:
                # Linear interpolation
                alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                s0, s1 = self.states[i], self.states[i + 1]
                
                # Interpolate position and angle
                pos = (1 - alpha) * s0.position + alpha * s1.position
                angle = self._interpolate_angle(s0.angle, s1.angle, alpha)
                vel = (1 - alpha) * s0.velocity + alpha * s1.velocity
                
                return State(
                    time_stamp=t,
                    position=pos,
                    angle=angle,
                    velocity=vel
                )
        
        return None
    
    @staticmethod
    def _interpolate_angle(a0: float, a1: float, alpha: float) -> float:
        """Interpolate angle handling wraparound."""
        diff = a1 - a0
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return a0 + alpha * diff
    
    @property
    def time_range(self) -> Tuple[float, float]:
        """Return (start_time, end_time) of the trajectory."""
        if not self.states:
            return (0.0, 0.0)
        return (self.states[0].time_stamp, self.states[-1].time_stamp)
