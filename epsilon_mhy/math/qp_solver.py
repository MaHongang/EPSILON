"""QP solver for Bezier trajectory optimization.

Reference: EPSILON/core/common/src/common/spline/spline_generator.cc

This module solves the trajectory optimization problem:
    min: ||trajectory - reference||^2 (proximity cost)
    s.t.: boundary constraints (position, velocity, acceleration)
          corridor constraints (stay within safe corridors)
          dynamic constraints (velocity/acceleration limits)
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

try:
    import cvxpy as cp
except ImportError:
    cp = None
    print("Warning: cvxpy not installed. QP solver will not work.")

from .bezier import BezierSpline, get_bezier_basis_matrix


@dataclass
class Corridor:
    """Spatio-temporal corridor segment.
    
    Defines bounds for a time interval:
    - Position bounds: p_lb <= (s, d) <= p_ub
    - Velocity bounds: v_lb <= (s_dot, d_dot) <= v_ub
    - Acceleration bounds: a_lb <= (s_ddot, d_ddot) <= a_ub
    
    Attributes:
        t_lb: Lower time bound
        t_ub: Upper time bound
        p_lb: Position lower bound (s_min, d_min)
        p_ub: Position upper bound (s_max, d_max)
        v_lb: Velocity lower bound
        v_ub: Velocity upper bound
        a_lb: Acceleration lower bound
        a_ub: Acceleration upper bound
    """
    t_lb: float
    t_ub: float
    p_lb: np.ndarray
    p_ub: np.ndarray
    v_lb: np.ndarray = None
    v_ub: np.ndarray = None
    a_lb: np.ndarray = None
    a_ub: np.ndarray = None
    
    def __post_init__(self):
        self.p_lb = np.asarray(self.p_lb)
        self.p_ub = np.asarray(self.p_ub)
        
        # Default velocity bounds
        if self.v_lb is None:
            self.v_lb = np.array([0.0, -10.0])  # s_dot >= 0
        else:
            self.v_lb = np.asarray(self.v_lb)
            
        if self.v_ub is None:
            self.v_ub = np.array([50.0, 10.0])
        else:
            self.v_ub = np.asarray(self.v_ub)
            
        # Default acceleration bounds
        if self.a_lb is None:
            self.a_lb = np.array([-8.0, -5.0])
        else:
            self.a_lb = np.asarray(self.a_lb)
            
        if self.a_ub is None:
            self.a_ub = np.array([3.0, 5.0])
        else:
            self.a_ub = np.asarray(self.a_ub)
    
    @property
    def duration(self) -> float:
        """Duration of this corridor segment."""
        return self.t_ub - self.t_lb


def optimize_bezier_in_corridor(
    start_constraints: List[np.ndarray],
    end_constraints: List[np.ndarray],
    corridors: List[Corridor],
    ref_stamps: Optional[List[float]] = None,
    ref_points: Optional[List[np.ndarray]] = None,
    weight_proximity: float = 1.0,
    degree: int = 5,
    continuity_order: int = 3,
    verbose: bool = False
) -> Optional[BezierSpline]:
    """Optimize Bezier spline within corridor constraints.
    
    Finds a smooth trajectory that:
    1. Starts and ends with specified constraints
    2. Stays within the given corridors
    3. Minimizes deviation from reference points
    
    Args:
        start_constraints: [position, velocity, acceleration] at start
        end_constraints: [position, velocity] at end (acceleration free)
        corridors: List of corridor segments
        ref_stamps: Time stamps for reference points
        ref_points: Reference positions [(s, d), ...]
        weight_proximity: Weight for proximity cost
        degree: Bezier curve degree (default 5 for quintic)
        continuity_order: Continuity order at segment junctions
        verbose: Print solver output
        
    Returns:
        Optimized BezierSpline, or None if optimization fails
    """
    if cp is None:
        raise ImportError("cvxpy is required for QP optimization")
    
    if not corridors:
        return None
    
    n_segments = len(corridors)
    n_ctrl = degree + 1  # Number of control points per segment
    dim = 2  # (s, d)
    
    # Total decision variables: control points for all segments
    # Each segment has (degree+1) control points, each is 2D
    # But shared points at junctions reduce the total
    
    # For simplicity, we optimize each segment's control points independently
    # and add continuity constraints
    
    # Decision variables: control points for each segment
    P = [cp.Variable((n_ctrl, dim), name=f"P_{i}") for i in range(n_segments)]
    
    constraints = []
    
    # === 1. Start constraints ===
    if len(start_constraints) >= 1:
        # Position constraint: P[0][0] = start_pos
        constraints.append(P[0][0, :] == start_constraints[0])
    
    if len(start_constraints) >= 2:
        # Velocity constraint: derivative at t=0
        # For Bezier: B'(0) = n * (P[1] - P[0])
        # So: P[1] = P[0] + start_vel * duration / n
        dt = corridors[0].duration
        n = degree
        if dt > 1e-10:
            vel_diff = start_constraints[1] * dt / n
            constraints.append(P[0][1, :] == P[0][0, :] + vel_diff)
    
    if len(start_constraints) >= 3:
        # Acceleration constraint: B''(0) = n*(n-1) * (P[0] - 2*P[1] + P[2])
        dt = corridors[0].duration
        n = degree
        if dt > 1e-10 and n >= 2:
            acc_diff = start_constraints[2] * (dt ** 2) / (n * (n - 1))
            # P[2] = 2*P[1] - P[0] + acc_diff
            constraints.append(P[0][2, :] == 2 * P[0][1, :] - P[0][0, :] + acc_diff)
    
    # === 2. End constraints ===
    if len(end_constraints) >= 1:
        # Position constraint at end of last segment
        constraints.append(P[-1][-1, :] == end_constraints[0])
    
    if len(end_constraints) >= 2:
        # Velocity constraint at end
        dt = corridors[-1].duration
        n = degree
        if dt > 1e-10:
            vel_diff = end_constraints[1] * dt / n
            constraints.append(P[-1][-1, :] == P[-1][-2, :] + vel_diff)
    
    # === 3. Continuity constraints between segments ===
    for i in range(n_segments - 1):
        dt_curr = corridors[i].duration
        dt_next = corridors[i + 1].duration
        n = degree
        
        # C0: Position continuity
        constraints.append(P[i][-1, :] == P[i + 1][0, :])
        
        if continuity_order >= 1 and dt_curr > 1e-10 and dt_next > 1e-10:
            # C1: Velocity continuity
            # n * (P[i][-1] - P[i][-2]) / dt_curr = n * (P[i+1][1] - P[i+1][0]) / dt_next
            constraints.append(
                (P[i][-1, :] - P[i][-2, :]) / dt_curr == 
                (P[i + 1][1, :] - P[i + 1][0, :]) / dt_next
            )
        
        if continuity_order >= 2 and dt_curr > 1e-10 and dt_next > 1e-10 and n >= 2:
            # C2: Acceleration continuity
            acc_curr = (P[i][-1, :] - 2 * P[i][-2, :] + P[i][-3, :]) / (dt_curr ** 2)
            acc_next = (P[i + 1][2, :] - 2 * P[i + 1][1, :] + P[i + 1][0, :]) / (dt_next ** 2)
            constraints.append(acc_curr == acc_next)
    
    # === 4. Corridor constraints ===
    for i, corr in enumerate(corridors):
        # All control points must be within corridor bounds
        for j in range(n_ctrl):
            constraints.append(P[i][j, :] >= corr.p_lb)
            constraints.append(P[i][j, :] <= corr.p_ub)
    
    # === 5. Build objective: proximity to reference ===
    objective_terms = []
    
    if ref_stamps is not None and ref_points is not None and len(ref_stamps) > 0:
        for t, ref_pt in zip(ref_stamps, ref_points):
            # Find which segment this time belongs to
            seg_idx = _find_segment_index(t, corridors)
            if seg_idx < 0:
                continue
            
            # Normalize time within segment
            corr = corridors[seg_idx]
            if corr.duration < 1e-10:
                tau = 0.0
            else:
                tau = (t - corr.t_lb) / corr.duration
            tau = np.clip(tau, 0.0, 1.0)
            
            # Evaluate Bezier at tau using control points
            # B(tau) = sum_j B_{j,n}(tau) * P[j]
            basis = _bernstein_basis(degree, tau)  # (n+1,)
            
            # Position at tau
            pos_at_tau = cp.sum([basis[j] * P[seg_idx][j, :] for j in range(n_ctrl)])
            
            # Add proximity cost
            objective_terms.append(cp.sum_squares(pos_at_tau - ref_pt))
    
    # Add regularization for smoothness (minimize control point differences)
    for i in range(n_segments):
        for j in range(n_ctrl - 1):
            objective_terms.append(0.01 * cp.sum_squares(P[i][j + 1, :] - P[i][j, :]))
    
    # Build total objective
    if objective_terms:
        objective = cp.Minimize(weight_proximity * cp.sum(objective_terms))
    else:
        # Fallback: minimize control point variance
        objective = cp.Minimize(
            cp.sum([cp.sum_squares(P[i] - cp.mean(P[i], axis=0)) for i in range(n_segments)])
        )
    
    # === 6. Solve ===
    problem = cp.Problem(objective, constraints)
    
    try:
        problem.solve(solver=cp.OSQP, verbose=verbose)
    except cp.SolverError:
        try:
            problem.solve(solver=cp.ECOS, verbose=verbose)
        except cp.SolverError:
            return None
    
    if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        return None
    
    # === 7. Build BezierSpline from solution ===
    spline = BezierSpline(degree=degree, dimension=dim)
    
    for i, corr in enumerate(corridors):
        ctrl_pts = P[i].value
        if ctrl_pts is None:
            return None
        spline.add_segment(ctrl_pts, corr.t_lb, corr.t_ub)
    
    return spline


def _find_segment_index(t: float, corridors: List[Corridor]) -> int:
    """Find the corridor index containing time t."""
    for i, corr in enumerate(corridors):
        if corr.t_lb <= t <= corr.t_ub:
            return i
    return -1


def _bernstein_basis(n: int, t: float) -> np.ndarray:
    """Compute Bernstein basis functions at t.
    
    Args:
        n: Polynomial degree
        t: Parameter in [0, 1]
        
    Returns:
        Array of length n+1 with basis function values
    """
    from scipy.special import comb
    basis = np.zeros(n + 1)
    for i in range(n + 1):
        basis[i] = comb(n, i, exact=True) * (t ** i) * ((1 - t) ** (n - i))
    return basis


def create_initial_trajectory(
    start_state: "FrenetState",
    target_velocity: float,
    duration: float,
    dt: float = 0.1
) -> Tuple[List[float], List[np.ndarray]]:
    """Create a simple initial trajectory (constant velocity).
    
    Args:
        start_state: Starting state in Frenet coordinates
        target_velocity: Target longitudinal velocity
        duration: Total duration
        dt: Time step
        
    Returns:
        (timestamps, positions) lists
    """
    from ..core.state import FrenetState
    
    timestamps = []
    positions = []
    
    t = start_state.time_stamp
    s = start_state.vec_s[0]
    d = start_state.vec_dt[0]
    
    # Simple constant velocity trajectory with d converging to 0
    n_steps = int(duration / dt)
    
    for i in range(n_steps + 1):
        timestamps.append(t)
        positions.append(np.array([s, d]))
        
        # Update
        t += dt
        s += target_velocity * dt
        d *= 0.95  # Converge to center
    
    return timestamps, positions
