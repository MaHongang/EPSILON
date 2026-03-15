"""Lane/Reference path representation.

Reference: EPSILON/core/common/inc/common/lane/lane.h
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
from scipy.interpolate import CubicSpline as ScipyCubicSpline


@dataclass
class Lane:
    """Reference lane/path representation.
    
    A lane is defined by a sequence of waypoints that are interpolated
    using cubic splines. It provides methods for:
    - Getting position, tangent, curvature at any arc length
    - Finding the closest point on the lane to a given position
    
    Attributes:
        waypoints: Array of shape (N, 2) containing (x, y) positions
        arc_lengths: Cumulative arc lengths at each waypoint
    """
    waypoints: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    arc_lengths: np.ndarray = field(default_factory=lambda: np.zeros(0))
    
    _spline_x: Optional[ScipyCubicSpline] = field(default=None, repr=False)
    _spline_y: Optional[ScipyCubicSpline] = field(default=None, repr=False)
    
    def __post_init__(self):
        self.waypoints = np.asarray(self.waypoints, dtype=np.float64)
        if len(self.waypoints) > 1:
            self._compute_arc_lengths()
            self._build_splines()
    
    @classmethod
    def from_points(cls, points: np.ndarray) -> "Lane":
        """Create a lane from waypoints.
        
        Args:
            points: Array of shape (N, 2) with (x, y) coordinates
        """
        lane = cls(waypoints=np.asarray(points, dtype=np.float64))
        return lane
    
    @classmethod
    def from_function(cls, func, s_range: Tuple[float, float], 
                     num_points: int = 100) -> "Lane":
        """Create a lane from a parametric function.
        
        Args:
            func: Function that takes arc length s and returns (x, y)
            s_range: (s_min, s_max) range
            num_points: Number of sample points
        """
        s_values = np.linspace(s_range[0], s_range[1], num_points)
        points = np.array([func(s) for s in s_values])
        return cls.from_points(points)
    
    def _compute_arc_lengths(self):
        """Compute cumulative arc lengths from waypoints."""
        diffs = np.diff(self.waypoints, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        self.arc_lengths = np.zeros(len(self.waypoints))
        self.arc_lengths[1:] = np.cumsum(segment_lengths)
    
    def _build_splines(self):
        """Build cubic spline interpolators."""
        if len(self.waypoints) < 2:
            return
        
        self._spline_x = ScipyCubicSpline(
            self.arc_lengths, self.waypoints[:, 0], bc_type='natural'
        )
        self._spline_y = ScipyCubicSpline(
            self.arc_lengths, self.waypoints[:, 1], bc_type='natural'
        )
    
    @property
    def begin(self) -> float:
        """Start arc length."""
        return self.arc_lengths[0] if len(self.arc_lengths) > 0 else 0.0
    
    @property
    def end(self) -> float:
        """End arc length."""
        return self.arc_lengths[-1] if len(self.arc_lengths) > 0 else 0.0
    
    @property 
    def length(self) -> float:
        """Total length of the lane."""
        return self.end - self.begin
    
    def get_position(self, s: float) -> np.ndarray:
        """Get (x, y) position at arc length s."""
        if self._spline_x is None or self._spline_y is None:
            return np.zeros(2)
        
        s = np.clip(s, self.begin, self.end)
        return np.array([self._spline_x(s), self._spline_y(s)])
    
    def get_tangent(self, s: float) -> np.ndarray:
        """Get unit tangent vector at arc length s."""
        if self._spline_x is None or self._spline_y is None:
            return np.array([1.0, 0.0])
        
        s = np.clip(s, self.begin, self.end)
        dx = self._spline_x(s, 1)  # First derivative
        dy = self._spline_y(s, 1)
        
        length = np.sqrt(dx**2 + dy**2)
        if length < 1e-10:
            return np.array([1.0, 0.0])
        
        return np.array([dx, dy]) / length
    
    def get_angle(self, s: float) -> float:
        """Get heading angle at arc length s."""
        tangent = self.get_tangent(s)
        return np.arctan2(tangent[1], tangent[0])
    
    def get_curvature(self, s: float) -> float:
        """Get curvature at arc length s.
        
        Curvature = (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)
        """
        if self._spline_x is None or self._spline_y is None:
            return 0.0
        
        s = np.clip(s, self.begin, self.end)
        dx = self._spline_x(s, 1)
        dy = self._spline_y(s, 1)
        ddx = self._spline_x(s, 2)
        ddy = self._spline_y(s, 2)
        
        denom = (dx**2 + dy**2) ** 1.5
        if denom < 1e-10:
            return 0.0
        
        return (dx * ddy - dy * ddx) / denom
    
    def get_normal(self, s: float) -> np.ndarray:
        """Get unit normal vector at arc length s (pointing left)."""
        tangent = self.get_tangent(s)
        return np.array([-tangent[1], tangent[0]])
    
    def find_closest_point(self, position: np.ndarray, 
                          s_hint: Optional[float] = None) -> Tuple[float, float]:
        """Find the closest point on the lane to a given position.
        
        Args:
            position: (x, y) position to project
            s_hint: Optional initial guess for arc length
            
        Returns:
            (s, d): Arc length of closest point and signed lateral distance
                   (positive = left of lane, negative = right)
        """
        position = np.asarray(position)
        
        # Initial search: sample along the lane
        if s_hint is not None:
            # Search in neighborhood of hint
            s_range = np.linspace(
                max(self.begin, s_hint - 20),
                min(self.end, s_hint + 20),
                100
            )
        else:
            s_range = np.linspace(self.begin, self.end, 200)
        
        # Find minimum distance point
        min_dist = float('inf')
        best_s = self.begin
        
        for s in s_range:
            point = self.get_position(s)
            dist = np.linalg.norm(position - point)
            if dist < min_dist:
                min_dist = dist
                best_s = s
        
        # Refine with Newton's method
        for _ in range(5):
            p = self.get_position(best_s)
            tangent = self.get_tangent(best_s)
            diff = position - p
            ds = np.dot(diff, tangent)
            best_s = np.clip(best_s + ds, self.begin, self.end)
            if abs(ds) < 1e-6:
                break
        
        # Compute signed lateral distance
        p = self.get_position(best_s)
        normal = self.get_normal(best_s)
        d = np.dot(position - p, normal)
        
        return best_s, d
    
    def get_frenet_point(self, position: np.ndarray,
                        s_hint: Optional[float] = None) -> Tuple[float, float]:
        """Convert Cartesian point to Frenet coordinates.
        
        Args:
            position: (x, y) in Cartesian coordinates
            
        Returns:
            (s, d): Frenet coordinates
        """
        return self.find_closest_point(position, s_hint)
    
    def get_cartesian_point(self, s: float, d: float) -> np.ndarray:
        """Convert Frenet point to Cartesian coordinates.
        
        Args:
            s: Arc length
            d: Lateral offset (positive = left)
            
        Returns:
            (x, y): Cartesian position
        """
        p = self.get_position(s)
        n = self.get_normal(s)
        return p + d * n
