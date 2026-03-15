#!/usr/bin/env python3
"""Simple example of using the SSC planner.

This example demonstrates basic usage of the SSC planner with a straight road
and no obstacles.
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from epsilon_mhy.core.state import State
from epsilon_mhy.core.lane import Lane
from epsilon_mhy.planning.ssc_planner import SscPlanner, SscPlannerConfig, create_straight_lane


def main():
    print("=" * 60)
    print("SSC Planner - Simple Example (No Obstacles)")
    print("=" * 60)
    
    # Create a straight reference lane
    lane = create_straight_lane(length=200.0, num_points=100)
    print(f"Created straight lane: length = {lane.length:.1f}m")
    
    # Create initial ego state
    initial_state = State(
        time_stamp=0.0,
        position=np.array([10.0, 0.5]),  # Start at x=10, y=0.5 (slight offset)
        angle=0.0,
        velocity=10.0,
        acceleration=0.0
    )
    print(f"Initial state: pos=({initial_state.x:.1f}, {initial_state.y:.1f}), "
          f"v={initial_state.velocity:.1f} m/s")
    
    # Create planner with default configuration
    config = SscPlannerConfig(
        planning_horizon=5.0,
        target_velocity=15.0
    )
    planner = SscPlanner(config)
    
    # Plan trajectory (no obstacles)
    print("\nPlanning trajectory...")
    result = planner.plan(
        initial_state=initial_state,
        reference_lane=lane,
        obstacle_trajectories=[],
        target_velocity=15.0
    )
    
    print(f"\nPlanning result:")
    print(f"  Success: {result.success}")
    print(f"  Message: {result.message}")
    print(f"  Time cost: {result.time_cost:.2f} ms")
    
    if result.success:
        print(f"  Number of Frenet states: {len(result.frenet_states)}")
        print(f"  Number of Cartesian states: {len(result.cartesian_states)}")
        
        if result.trajectory:
            print(f"  Trajectory time range: [{result.trajectory.begin:.2f}, "
                  f"{result.trajectory.end:.2f}] s")
        
        # Print some trajectory samples
        print("\nTrajectory samples (Cartesian):")
        for i in range(0, len(result.cartesian_states), 10):
            s = result.cartesian_states[i]
            print(f"  t={s.time_stamp:.2f}s: pos=({s.x:.2f}, {s.y:.2f}), "
                  f"v={s.velocity:.2f} m/s")
        
        # Print last state
        if result.cartesian_states:
            last = result.cartesian_states[-1]
            print(f"\nFinal state:")
            print(f"  Position: ({last.x:.2f}, {last.y:.2f})")
            print(f"  Velocity: {last.velocity:.2f} m/s")
            print(f"  Time: {last.time_stamp:.2f} s")
    
    # Visualization (optional)
    try:
        from epsilon_mhy.utils.visualization import quick_plot_result
        import matplotlib.pyplot as plt
        
        print("\nGenerating visualization...")
        figures = quick_plot_result(result, lane, initial_state, [])
        plt.show()
    except ImportError:
        print("\nVisualization skipped (matplotlib not available)")
    
    return result.success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
