"""Bezier spline utilities.

Reference: EPSILON/core/common/inc/common/spline/bezier.h

Bezier curves are defined by control points. An n-th degree Bezier curve 
has n+1 control points and is parameterized by t in [0, 1].

For trajectory planning, we use Bezier splines which are piecewise Bezier 
curves with continuity constraints at the junctions.
"""

from typing import List, Tuple, Optional
import numpy as np
from scipy.special import comb


def bernstein_poly(n: int, i: int, t: float) -> float:
    """Compute Bernstein polynomial basis function.
    
    B_{i,n}(t) = C(n,i) * t^i * (1-t)^(n-i)
    
    Args:
        n: Polynomial degree
        i: Basis function index (0 <= i <= n)
        t: Parameter in [0, 1]
        
    Returns:
        Value of the basis function
    """
    return comb(n, i, exact=True) * (t ** i) * ((1 - t) ** (n - i))


def bernstein_matrix(n: int, t_values: np.ndarray) -> np.ndarray:
    """Compute Bernstein polynomial matrix.
    
    Args:
        n: Polynomial degree
        t_values: Array of parameter values
        
    Returns:
        Matrix of shape (len(t_values), n+1) where B[j, i] = B_{i,n}(t_j)
    """
    m = len(t_values)
    B = np.zeros((m, n + 1))
    for j, t in enumerate(t_values):
        for i in range(n + 1):
            B[j, i] = bernstein_poly(n, i, t)
    return B


class BezierCurve:
    """Single Bezier curve segment.
    
    A degree-n Bezier curve is defined by n+1 control points.
    """
    
    def __init__(self, control_points: np.ndarray, t_start: float = 0.0, 
                 t_end: float = 1.0):
        """Initialize Bezier curve.
        
        Args:
            control_points: Array of shape (n+1, dim) for degree-n curve
            t_start: Start parameter (mapped from 0)
            t_end: End parameter (mapped from 1)
        """
        self.control_points = np.asarray(control_points, dtype=np.float64)
        self.t_start = t_start
        self.t_end = t_end
        self.degree = len(control_points) - 1
        self.dimension = control_points.shape[1] if len(control_points.shape) > 1 else 1
        
        # Precompute derivative control points
        self._derivative_points: List[np.ndarray] = [self.control_points]
        self._compute_derivative_points()
    
    def _compute_derivative_points(self):
        """Precompute control points for derivative curves."""
        n = self.degree
        pts = self.control_points
        
        for order in range(1, self.degree + 1):
            n_curr = n - order + 1
            new_pts = np.zeros((n_curr, self.dimension))
            for i in range(n_curr):
                new_pts[i] = n_curr * (self._derivative_points[-1][i + 1] - 
                                       self._derivative_points[-1][i])
            self._derivative_points.append(new_pts)
    
    def _normalize_t(self, t: float) -> float:
        """Map external t to internal [0, 1] range."""
        if abs(self.t_end - self.t_start) < 1e-10:
            return 0.0
        return (t - self.t_start) / (self.t_end - self.t_start)
    
    def evaluate(self, t: float) -> np.ndarray:
        """Evaluate curve at parameter t.
        
        Args:
            t: Parameter value in [t_start, t_end]
            
        Returns:
            Position at t
        """
        tau = self._normalize_t(t)
        tau = np.clip(tau, 0.0, 1.0)
        return self._de_casteljau(tau, self.control_points)
    
    def derivative(self, t: float, order: int = 1) -> np.ndarray:
        """Evaluate derivative at parameter t.
        
        Args:
            t: Parameter value
            order: Derivative order
            
        Returns:
            Derivative value at t
        """
        if order > self.degree:
            return np.zeros(self.dimension)
        
        tau = self._normalize_t(t)
        tau = np.clip(tau, 0.0, 1.0)
        
        # Evaluate using derivative control points
        pts = self._derivative_points[order]
        value = self._de_casteljau(tau, pts)
        
        # Scale for parameter transformation
        scale = 1.0 / (self.t_end - self.t_start) ** order
        return value * scale
    
    def _de_casteljau(self, t: float, points: np.ndarray) -> np.ndarray:
        """De Casteljau's algorithm for Bezier evaluation."""
        n = len(points) - 1
        if n < 0:
            return np.zeros(self.dimension)
        
        # Work array
        pts = points.copy()
        for r in range(n):
            for i in range(n - r):
                pts[i] = (1 - t) * pts[i] + t * pts[i + 1]
        
        return pts[0]
    
    @property
    def start_point(self) -> np.ndarray:
        """First control point (curve starts here)."""
        return self.control_points[0]
    
    @property
    def end_point(self) -> np.ndarray:
        """Last control point (curve ends here)."""
        return self.control_points[-1]


class BezierSpline:
    """Piecewise Bezier spline.
    
    A spline composed of multiple Bezier curve segments with continuity
    at the junctions.
    """
    
    def __init__(self, degree: int = 5, dimension: int = 2):
        """Initialize Bezier spline.
        
        Args:
            degree: Degree of each Bezier segment
            dimension: Output dimension (default 2 for (s, d))
        """
        self.degree = degree
        self.dimension = dimension
        self.segments: List[BezierCurve] = []
    
    def add_segment(self, control_points: np.ndarray, 
                   t_start: float, t_end: float):
        """Add a Bezier segment.
        
        Args:
            control_points: Control points of shape (degree+1, dimension)
            t_start: Start time of segment
            t_end: End time of segment
        """
        curve = BezierCurve(control_points, t_start, t_end)
        self.segments.append(curve)
    
    @classmethod
    def from_control_points_list(cls, control_points_list: List[np.ndarray],
                                  time_breaks: List[float]) -> "BezierSpline":
        """Create spline from list of control points.
        
        Args:
            control_points_list: List of control point arrays
            time_breaks: Time values at segment boundaries [t0, t1, ..., tn]
                        where n = len(control_points_list)
        """
        if len(time_breaks) != len(control_points_list) + 1:
            raise ValueError("time_breaks should have one more element than control_points_list")
        
        degree = len(control_points_list[0]) - 1
        dimension = control_points_list[0].shape[1]
        
        spline = cls(degree=degree, dimension=dimension)
        for i, pts in enumerate(control_points_list):
            spline.add_segment(pts, time_breaks[i], time_breaks[i + 1])
        
        return spline
    
    @property
    def begin(self) -> float:
        """Start time."""
        if not self.segments:
            return 0.0
        return self.segments[0].t_start
    
    @property
    def end(self) -> float:
        """End time."""
        if not self.segments:
            return 0.0
        return self.segments[-1].t_end
    
    def evaluate(self, t: float) -> np.ndarray:
        """Evaluate spline at time t.
        
        Args:
            t: Time value
            
        Returns:
            Position (s, d) at time t
        """
        segment = self._find_segment(t)
        if segment is None:
            return np.zeros(self.dimension)
        return segment.evaluate(t)
    
    def derivative(self, t: float, order: int = 1) -> np.ndarray:
        """Evaluate derivative at time t.
        
        Args:
            t: Time value
            order: Derivative order
            
        Returns:
            Derivative value at time t
        """
        segment = self._find_segment(t)
        if segment is None:
            return np.zeros(self.dimension)
        return segment.derivative(t, order)
    
    def _find_segment(self, t: float) -> Optional[BezierCurve]:
        """Find the segment containing time t."""
        for seg in self.segments:
            if seg.t_start <= t <= seg.t_end:
                return seg
        # Clamp to endpoints
        if t < self.begin and self.segments:
            return self.segments[0]
        if t > self.end and self.segments:
            return self.segments[-1]
        return None
    
    def get_position(self, t: float) -> np.ndarray:
        """Alias for evaluate."""
        return self.evaluate(t)
    
    def get_velocity(self, t: float) -> np.ndarray:
        """Get velocity at time t."""
        return self.derivative(t, 1)
    
    def get_acceleration(self, t: float) -> np.ndarray:
        """Get acceleration at time t."""
        return self.derivative(t, 2)
    
    @property
    def num_segments(self) -> int:
        """Number of segments."""
        return len(self.segments)
    
    @property
    def total_control_points(self) -> int:
        """Total number of control points."""
        return sum(len(seg.control_points) for seg in self.segments)


def get_bezier_basis_matrix(degree: int) -> np.ndarray:
    """Get the matrix converting Bezier control points to polynomial coefficients.
    
    For a Bezier curve B(t) = sum_i P_i * B_{i,n}(t), this returns M such that
    B(t) = [1, t, t^2, ..., t^n] @ M @ [P_0, ..., P_n]^T
    
    Args:
        degree: Bezier curve degree
        
    Returns:
        Matrix of shape (degree+1, degree+1)
    """
    n = degree
    M = np.zeros((n + 1, n + 1))
    
    for j in range(n + 1):  # Power of t
        for i in range(n + 1):  # Control point index
            if j <= i:
                M[j, i] = ((-1) ** (i - j)) * comb(n, i, exact=True) * comb(i, j, exact=True)
    
    return M
