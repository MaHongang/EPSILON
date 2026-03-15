"""
epsilon_mhy: Python implementation of SSC (Spatio-temporal Semantic Corridor) Planner

A trajectory planning library for autonomous vehicles with dynamic obstacle avoidance.
"""

from .core.state import State, FrenetState
from .core.vehicle import Vehicle, VehicleParam
from .core.lane import Lane
from .planning.ssc_planner import SscPlanner, SscPlannerConfig

__version__ = "0.1.0"
__all__ = [
    "State",
    "FrenetState",
    "Vehicle",
    "VehicleParam",
    "Lane",
    "SscPlanner",
    "SscPlannerConfig",
]
