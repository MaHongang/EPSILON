"""State representations for vehicles.

Reference: EPSILON/core/common/inc/common/state/state.h
           EPSILON/core/common/inc/common/state/frenet_state.h
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .types import EPS


@dataclass
class State:
    """Vehicle state in Cartesian coordinates.
    
    Attributes:
        time_stamp: Time stamp in seconds
        position: Position (x, y) in meters
        angle: Heading angle in radians
        curvature: Path curvature in 1/m
        velocity: Speed in m/s
        acceleration: Acceleration in m/s^2
        steer: Steering angle in radians
    """
    time_stamp: float = 0.0
    position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    angle: float = 0.0
    curvature: float = 0.0
    velocity: float = 0.0
    acceleration: float = 0.0
    steer: float = 0.0
    
    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float64)
        if self.position.shape != (2,):
            raise ValueError("position must be a 2D vector")
    
    @property
    def x(self) -> float:
        return self.position[0]
    
    @property
    def y(self) -> float:
        return self.position[1]
    
    def to_xy_theta(self) -> np.ndarray:
        """Return (x, y, theta) array."""
        return np.array([self.position[0], self.position[1], self.angle])
    
    def copy(self) -> "State":
        """Create a deep copy of this state."""
        return State(
            time_stamp=self.time_stamp,
            position=self.position.copy(),
            angle=self.angle,
            curvature=self.curvature,
            velocity=self.velocity,
            acceleration=self.acceleration,
            steer=self.steer
        )
    
    def __repr__(self) -> str:
        return (f"State(t={self.time_stamp:.3f}, pos=({self.x:.2f}, {self.y:.2f}), "
                f"θ={np.degrees(self.angle):.1f}°, v={self.velocity:.2f})")


@dataclass
class FrenetState:
    """Vehicle state in Frenet coordinates.
    
    In Frenet frame:
    - s: longitudinal position along the reference path
    - d: lateral deviation from the reference path
    
    Two derivative modes:
    - vec_dt: d derivatives w.r.t. time (high speed mode)
    - vec_ds: d derivatives w.r.t. arc length s (low speed mode)
    
    Attributes:
        time_stamp: Time stamp in seconds
        vec_s: (s, ds/dt, d²s/dt²) - longitudinal state
        vec_dt: (d, dd/dt, d²d/dt²) - lateral state w.r.t. time
        vec_ds: (d, dd/ds, d²d/ds²) - lateral state w.r.t. arc length
    """
    time_stamp: float = 0.0
    vec_s: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vec_dt: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vec_ds: np.ndarray = field(default_factory=lambda: np.zeros(3))
    is_ds_usable: bool = True
    
    def __post_init__(self):
        self.vec_s = np.asarray(self.vec_s, dtype=np.float64)
        self.vec_dt = np.asarray(self.vec_dt, dtype=np.float64)
        self.vec_ds = np.asarray(self.vec_ds, dtype=np.float64)
    
    @property
    def s(self) -> float:
        """Longitudinal position."""
        return self.vec_s[0]
    
    @property
    def s_dot(self) -> float:
        """Longitudinal velocity."""
        return self.vec_s[1]
    
    @property
    def s_ddot(self) -> float:
        """Longitudinal acceleration."""
        return self.vec_s[2]
    
    @property
    def d(self) -> float:
        """Lateral position."""
        return self.vec_dt[0]
    
    @property
    def d_dot(self) -> float:
        """Lateral velocity w.r.t. time."""
        return self.vec_dt[1]
    
    @property
    def d_ddot(self) -> float:
        """Lateral acceleration w.r.t. time."""
        return self.vec_dt[2]
    
    @classmethod
    def from_s_dt(cls, s: np.ndarray, dt: np.ndarray, 
                  time_stamp: float = 0.0) -> "FrenetState":
        """Initialize with s and d derivatives w.r.t. time.
        
        Computes ds derivatives from dt derivatives.
        """
        vec_s = np.asarray(s, dtype=np.float64)
        vec_dt = np.asarray(dt, dtype=np.float64)
        vec_ds = np.zeros(3)
        
        vec_ds[0] = vec_dt[0]  # d is the same
        
        if abs(vec_s[1]) > EPS:
            # d' = dd/ds = (dd/dt) / (ds/dt)
            vec_ds[1] = vec_dt[1] / vec_s[1]
            # d'' = (d²d/dt² - d' * d²s/dt²) / (ds/dt)²
            vec_ds[2] = (vec_dt[2] - vec_ds[1] * vec_s[2]) / (vec_s[1] ** 2)
            is_ds_usable = True
        else:
            vec_ds[1] = 0.0
            vec_ds[2] = 0.0
            is_ds_usable = False
        
        return cls(
            time_stamp=time_stamp,
            vec_s=vec_s,
            vec_dt=vec_dt,
            vec_ds=vec_ds,
            is_ds_usable=is_ds_usable
        )
    
    @classmethod
    def from_s_ds(cls, s: np.ndarray, ds: np.ndarray,
                  time_stamp: float = 0.0) -> "FrenetState":
        """Initialize with s and d derivatives w.r.t. arc length.
        
        Computes dt derivatives from ds derivatives.
        """
        vec_s = np.asarray(s, dtype=np.float64)
        vec_ds = np.asarray(ds, dtype=np.float64)
        vec_dt = np.zeros(3)
        
        vec_dt[0] = vec_ds[0]  # d is the same
        # dd/dt = dd/ds * ds/dt
        vec_dt[1] = vec_s[1] * vec_ds[1]
        # d²d/dt² = d²d/ds² * (ds/dt)² + dd/ds * d²s/dt²
        vec_dt[2] = vec_ds[2] * vec_s[1] ** 2 + vec_ds[1] * vec_s[2]
        
        return cls(
            time_stamp=time_stamp,
            vec_s=vec_s,
            vec_dt=vec_dt,
            vec_ds=vec_ds,
            is_ds_usable=True
        )
    
    def copy(self) -> "FrenetState":
        """Create a deep copy of this state."""
        return FrenetState(
            time_stamp=self.time_stamp,
            vec_s=self.vec_s.copy(),
            vec_dt=self.vec_dt.copy(),
            vec_ds=self.vec_ds.copy(),
            is_ds_usable=self.is_ds_usable
        )
    
    def __repr__(self) -> str:
        return (f"FrenetState(t={self.time_stamp:.3f}, "
                f"s=({self.vec_s[0]:.2f}, {self.vec_s[1]:.2f}, {self.vec_s[2]:.2f}), "
                f"d=({self.vec_dt[0]:.2f}, {self.vec_dt[1]:.2f}, {self.vec_dt[2]:.2f}))")
