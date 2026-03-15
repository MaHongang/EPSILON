"""Basic type definitions for epsilon_mhy."""

from enum import IntEnum
from typing import Tuple
import numpy as np


class ErrorType(IntEnum):
    """Error types matching the original C++ implementation."""
    SUCCESS = 0
    WRONG_STATUS = 1
    ILLEGAL_INPUT = 2
    UNKNOWN = 3


# Type aliases
Vec2f = np.ndarray  # shape (2,)
Vec3f = np.ndarray  # shape (3,)

# Constants
EPS = 1e-6
BIG_EPS = 1e-1
SMALL_EPS = 1e-10
PI = np.pi
INF = 1e20

INVALID_AGENT_ID = -1
INVALID_LANE_ID = -1
