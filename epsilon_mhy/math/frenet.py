"""Frenet coordinate transformation.

Reference: EPSILON/core/common/src/common/state/state_transformer.cc

The Frenet frame is defined by a reference path (lane). For a point:
- s: arc length along the reference path
- d: signed lateral deviation (positive = left)

For a state with velocity and acceleration:
- vec_s = [s, ds/dt, d²s/dt²]
- vec_d = [d, dd/dt, d²d/dt²]
"""

from typing import Optional, Tuple
import numpy as np

from ..core.state import State, FrenetState
from ..core.lane import Lane
from ..core.types import EPS


class StateTransformer:
    """Transform between Cartesian and Frenet coordinates.
    
    Given a reference lane, this class provides methods to:
    1. Transform a Cartesian State to FrenetState
    2. Transform a FrenetState back to Cartesian State
    3. Transform points between coordinate systems
    """
    
    def __init__(self, lane: Lane):
        """Initialize transformer with a reference lane.
        
        Args:
            lane: Reference lane defining the Frenet frame
        """
        self.lane = lane
    
    def global_to_frenet_point(self, position: np.ndarray,
                               s_hint: Optional[float] = None) -> Tuple[float, float]:
        """Transform a Cartesian point to Frenet coordinates.
        
        Args:
            position: (x, y) in global Cartesian coordinates
            s_hint: Optional hint for arc length (speeds up search)
            
        Returns:
            (s, d): Frenet coordinates
        """
        return self.lane.get_frenet_point(position, s_hint)
    
    def frenet_to_global_point(self, s: float, d: float) -> np.ndarray:
        """Transform a Frenet point to Cartesian coordinates.
        
        Args:
            s: Arc length along reference
            d: Lateral deviation
            
        Returns:
            (x, y): Global Cartesian position
        """
        return self.lane.get_cartesian_point(s, d)
    
    def global_to_frenet(self, state: State,
                        s_hint: Optional[float] = None) -> FrenetState:
        """Transform a Cartesian state to Frenet state.
        
        This transforms position, velocity, and acceleration to Frenet frame.
        
        Args:
            state: State in global Cartesian coordinates
            s_hint: Optional hint for arc length
            
        Returns:
            FrenetState in Frenet coordinates
        """
        # Find projection point
        s, d = self.global_to_frenet_point(state.position, s_hint)
        
        # Reference path properties at projection point
        ref_angle = self.lane.get_angle(s)
        ref_curvature = self.lane.get_curvature(s)
        
        # Heading difference
        delta_theta = self._normalize_angle(state.angle - ref_angle)
        
        # Velocity decomposition
        # v_s = v * cos(delta_theta) / (1 - curvature * d)
        # v_d = v * sin(delta_theta)
        one_minus_kd = 1.0 - ref_curvature * d
        if abs(one_minus_kd) < EPS:
            one_minus_kd = EPS
        
        cos_dt = np.cos(delta_theta)
        sin_dt = np.sin(delta_theta)
        
        v = state.velocity
        s_dot = v * cos_dt / one_minus_kd
        d_dot = v * sin_dt
        
        # Acceleration decomposition
        # This is a simplified version; full derivation involves curvature derivatives
        a = state.acceleration
        kappa = state.curvature  # Curvature of vehicle path
        
        # d_prime = dd/ds = tan(delta_theta) * (1 - kappa_ref * d)
        d_prime = np.tan(delta_theta) * one_minus_kd if abs(cos_dt) > EPS else 0.0
        
        # s_ddot = a * cos(delta_theta) / (1 - kappa_ref * d) + s_dot^2 * d * kappa_ref' / (1 - kappa_ref * d)
        # Simplified: assume kappa_ref is locally constant
        s_ddot = a * cos_dt / one_minus_kd + s_dot**2 * ref_curvature * d_prime / one_minus_kd
        
        # d_ddot = a * sin(delta_theta) - s_dot^2 * (kappa - kappa_ref) * (1 - kappa_ref * d) * cos(delta_theta)
        d_ddot = a * sin_dt
        
        vec_s = np.array([s, s_dot, s_ddot])
        vec_dt = np.array([d, d_dot, d_ddot])
        
        return FrenetState.from_s_dt(vec_s, vec_dt, state.time_stamp)
    
    def frenet_to_global(self, fs: FrenetState) -> State:
        """Transform a Frenet state to Cartesian state.
        
        Args:
            fs: State in Frenet coordinates
            
        Returns:
            State in global Cartesian coordinates
        """
        s = fs.vec_s[0]
        d = fs.vec_dt[0]
        s_dot = fs.vec_s[1]
        d_dot = fs.vec_dt[1]
        s_ddot = fs.vec_s[2]
        d_ddot = fs.vec_dt[2]
        
        # Reference path properties
        ref_pos = self.lane.get_position(s)
        ref_angle = self.lane.get_angle(s)
        ref_curvature = self.lane.get_curvature(s)
        ref_normal = self.lane.get_normal(s)
        
        # Position
        position = ref_pos + d * ref_normal
        
        # Heading angle
        # delta_theta = atan(d_dot / s_dot) when using time derivatives
        if abs(s_dot) > EPS:
            delta_theta = np.arctan2(d_dot, s_dot * (1 - ref_curvature * d))
        else:
            delta_theta = 0.0
        
        angle = ref_angle + delta_theta
        angle = self._normalize_angle(angle)
        
        # Velocity magnitude
        one_minus_kd = 1.0 - ref_curvature * d
        if abs(one_minus_kd) < EPS:
            one_minus_kd = EPS
            
        v = np.sqrt((s_dot * one_minus_kd)**2 + d_dot**2)
        
        # Acceleration (simplified)
        cos_dt = np.cos(delta_theta)
        sin_dt = np.sin(delta_theta)
        a = s_ddot * one_minus_kd * cos_dt + d_ddot * sin_dt
        
        # Curvature
        # κ = (κ_ref * cos(Δθ) + d''_s) / (1 - κ_ref * d) / cos(Δθ)
        # Using approximate formula for low curvature
        if v > EPS and abs(cos_dt) > EPS:
            # d'_s = d_dot / s_dot (derivative w.r.t. s)
            d_prime_s = d_dot / s_dot if abs(s_dot) > EPS else 0.0
            d_double_prime_s = 0.0  # Simplified
            curvature = (ref_curvature * cos_dt + d_double_prime_s) / one_minus_kd / cos_dt
        else:
            curvature = ref_curvature
        
        return State(
            time_stamp=fs.time_stamp,
            position=position,
            angle=angle,
            curvature=curvature,
            velocity=v,
            acceleration=a
        )
    
    def transform_vehicle_vertices(self, vertices: np.ndarray,
                                   s_hint: Optional[float] = None) -> np.ndarray:
        """Transform vehicle vertices from global to Frenet coordinates.
        
        Args:
            vertices: Array of shape (N, 2) with (x, y) positions
            s_hint: Optional arc length hint
            
        Returns:
            Array of shape (N, 2) with (s, d) positions
        """
        result = np.zeros_like(vertices)
        current_s = s_hint
        for i, vertex in enumerate(vertices):
            s, d = self.global_to_frenet_point(vertex, current_s)
            result[i] = [s, d]
            current_s = s  # Use as hint for next vertex
        return result
    
    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
