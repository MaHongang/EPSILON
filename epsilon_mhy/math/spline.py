"""Cubic spline utilities.

Reference: EPSILON/core/common/thirdparty/tk_spline/spline.h
"""

from typing import List, Tuple, Optional
import numpy as np
from scipy.interpolate import CubicSpline as ScipyCubicSpline


class CubicSpline:
    """Cubic spline interpolator.
    
    Wrapper around scipy's CubicSpline with convenient interface
    for trajectory planning applications.
    """
    
    def __init__(self, x: np.ndarray, y: np.ndarray, 
                 bc_type: str = 'natural'):
        """Initialize cubic spline.
        
        Args:
            x: Independent variable values (must be monotonically increasing)
            y: Dependent variable values (can be multi-dimensional)
            bc_type: Boundary condition type ('natural', 'clamped', 'not-a-knot')
        """
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        
        if len(self.x) < 2:
            raise ValueError("Need at least 2 points for spline interpolation")
        
        self._spline = ScipyCubicSpline(self.x, self.y, bc_type=bc_type)
    
    @property
    def begin(self) -> float:
        """Start of valid parameter range."""
        return self.x[0]
    
    @property
    def end(self) -> float:
        """End of valid parameter range."""
        return self.x[-1]
    
    def __call__(self, t: float, deriv: int = 0) -> np.ndarray:
        """Evaluate spline or its derivative.
        
        Args:
            t: Parameter value
            deriv: Derivative order (0=position, 1=velocity, 2=acceleration)
            
        Returns:
            Value at t
        """
        return self._spline(t, deriv)
    
    def evaluate(self, t: float) -> np.ndarray:
        """Evaluate spline at parameter t."""
        return self._spline(t)
    
    def derivative(self, t: float, order: int = 1) -> np.ndarray:
        """Evaluate derivative at parameter t."""
        return self._spline(t, order)
    
    def evaluate_array(self, t_array: np.ndarray) -> np.ndarray:
        """Evaluate spline at multiple parameter values."""
        return self._spline(t_array)


class PolynomialSpline:
    """Piecewise polynomial spline.
    
    A spline composed of polynomial segments, each defined by coefficients.
    """
    
    def __init__(self, degree: int, dimension: int = 2):
        """Initialize polynomial spline.
        
        Args:
            degree: Polynomial degree (e.g., 5 for quintic)
            dimension: Output dimension (default 2 for (s, d))
        """
        self.degree = degree
        self.dimension = dimension
        self.segments: List[PolynomialSegment] = []
    
    def add_segment(self, t_start: float, t_end: float, 
                   coeffs: np.ndarray):
        """Add a polynomial segment.
        
        Args:
            t_start: Start time of segment
            t_end: End time of segment
            coeffs: Coefficients of shape (dimension, degree+1)
                   coeffs[d, i] is the coefficient of t^i for dimension d
        """
        self.segments.append(PolynomialSegment(t_start, t_end, coeffs))
    
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
        """Evaluate spline at time t."""
        segment = self._find_segment(t)
        if segment is None:
            return np.zeros(self.dimension)
        return segment.evaluate(t)
    
    def derivative(self, t: float, order: int = 1) -> np.ndarray:
        """Evaluate derivative at time t."""
        segment = self._find_segment(t)
        if segment is None:
            return np.zeros(self.dimension)
        return segment.derivative(t, order)
    
    def _find_segment(self, t: float) -> Optional["PolynomialSegment"]:
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


class PolynomialSegment:
    """A single polynomial segment."""
    
    def __init__(self, t_start: float, t_end: float, coeffs: np.ndarray):
        """Initialize segment.
        
        Args:
            t_start: Start time
            t_end: End time
            coeffs: Coefficients of shape (dimension, degree+1)
        """
        self.t_start = t_start
        self.t_end = t_end
        self.coeffs = np.asarray(coeffs)
        self.dimension = self.coeffs.shape[0]
        self.degree = self.coeffs.shape[1] - 1
    
    def evaluate(self, t: float) -> np.ndarray:
        """Evaluate polynomial at time t."""
        tau = t - self.t_start
        result = np.zeros(self.dimension)
        for d in range(self.dimension):
            result[d] = np.polyval(self.coeffs[d, ::-1], tau)
        return result
    
    def derivative(self, t: float, order: int = 1) -> np.ndarray:
        """Evaluate derivative at time t."""
        tau = t - self.t_start
        result = np.zeros(self.dimension)
        for d in range(self.dimension):
            poly = np.poly1d(self.coeffs[d, ::-1])
            for _ in range(order):
                poly = poly.deriv()
            result[d] = poly(tau)
        return result
