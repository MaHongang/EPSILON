#!/usr/bin/env python3
"""Test SSC planner with multiple dynamic obstacle scenarios.

This script runs various trajectory planning scenarios and saves
the visualization results to the /result directory.

Scenarios:
1. Lane Change Overtake - passing a slower vehicle
2. Static Obstacle Avoidance - avoiding a stationary obstacle  
3. Multi-vehicle Avoidance - navigating between multiple vehicles
4. Crossing Obstacle - avoiding an obstacle crossing the path
"""

import numpy as np
import sys
import os
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from epsilon_mhy.core.state import State
from epsilon_mhy.core.vehicle import VehicleParam, ObstacleTrajectory
from epsilon_mhy.planning.ssc_planner import SscPlanner, SscPlannerConfig, create_straight_lane
from epsilon_mhy.planning.ssc_planner import PlanningResult
from epsilon_mhy.utils.visualization import Visualizer
from epsilon_mhy.core.lane import Lane

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving
import matplotlib.pyplot as plt


# Output directory
RESULT_DIR = "/home/mhy/manage_valet/EPSILON/result"


@dataclass
class ScenarioResult:
    """Result of a scenario test."""
    name: str
    success: bool
    time_cost: float
    message: str
    lateral_range: Tuple[float, float] = (0.0, 0.0)


def create_moving_obstacle(
    start_position: np.ndarray,
    velocity: np.ndarray,
    start_time: float = 0.0,
    duration: float = 10.0,
    dt: float = 0.1,
    width: float = 1.8,
    length: float = 4.5
) -> ObstacleTrajectory:
    """Create an obstacle moving with constant velocity."""
    states = []
    t = start_time
    pos = start_position.copy()
    
    while t <= start_time + duration:
        angle = np.arctan2(velocity[1], velocity[0]) if np.linalg.norm(velocity) > 0.1 else 0.0
        speed = np.linalg.norm(velocity)
        
        states.append(State(
            time_stamp=t,
            position=pos.copy(),
            angle=angle,
            velocity=speed
        ))
        
        pos += velocity * dt
        t += dt
    
    return ObstacleTrajectory.from_state_list(
        states=states,
        vehicle_param=VehicleParam(
            width=width,
            length=length,
            d_cr=1.2
        )
    )


def create_static_obstacle(
    position: np.ndarray,
    angle: float = 0.0,
    duration: float = 10.0,
    dt: float = 0.1,
    width: float = 1.8,
    length: float = 4.5
) -> ObstacleTrajectory:
    """Create a stationary obstacle."""
    states = []
    t = 0.0
    
    while t <= duration:
        states.append(State(
            time_stamp=t,
            position=position.copy(),
            angle=angle,
            velocity=0.0
        ))
        t += dt
    
    return ObstacleTrajectory.from_state_list(
        states=states,
        vehicle_param=VehicleParam(
            width=width,
            length=length,
            d_cr=1.2
        )
    )


def run_scenario(
    name: str,
    lane: Lane,
    initial_state: State,
    obstacles: List[ObstacleTrajectory],
    config: SscPlannerConfig,
    output_prefix: str
) -> ScenarioResult:
    """Run a single scenario and save visualizations."""
    
    print(f"\n{'='*60}")
    print(f"Scenario: {name}")
    print(f"{'='*60}")
    
    print(f"Ego: pos=({initial_state.x:.1f}, {initial_state.y:.1f}), v={initial_state.velocity:.1f} m/s")
    for i, obs in enumerate(obstacles):
        if obs.states:
            s = obs.states[0]
            print(f"Obstacle {i}: pos=({s.x:.1f}, {s.y:.1f}), v={s.velocity:.1f} m/s")
    
    # Create planner and run
    planner = SscPlanner(config)
    
    print("\nPlanning...")
    result = planner.plan(
        initial_state=initial_state,
        reference_lane=lane,
        obstacle_trajectories=obstacles,
        target_velocity=config.target_velocity
    )
    
    print(f"Result: {'Success' if result.success else 'Failed'}")
    print(f"Message: {result.message}")
    print(f"Time cost: {result.time_cost:.2f} ms")
    
    lateral_range = (0.0, 0.0)
    if result.success:
        print(f"Trajectory points: {len(result.cartesian_states)}")
        if result.frenet_states:
            d_vals = [fs.vec_dt[0] for fs in result.frenet_states]
            lateral_range = (min(d_vals), max(d_vals))
            print(f"Lateral deviation: [{lateral_range[0]:.2f}, {lateral_range[1]:.2f}] m")
    
    # Generate visualizations
    vis = Visualizer(figsize=(14, 8))
    
    # Plot scenario
    fig1 = vis.plot_scenario(
        lane=lane,
        ego_state=initial_state,
        obstacles=obstacles,
        trajectory_states=result.cartesian_states if result.success else None,
        title=f"{name}"
    )
    
    # Adjust axis limits for better visualization
    ax = fig1.axes[0]
    if result.success and result.cartesian_states:
        x_vals = [s.x for s in result.cartesian_states]
        y_vals = [s.y for s in result.cartesian_states]
        x_min, x_max = min(x_vals) - 10, max(x_vals) + 20
        y_min, y_max = min(y_vals) - 10, max(y_vals) + 10
        
        # Include obstacles in bounds
        for obs in obstacles:
            if obs.states:
                obs_x = [s.x for s in obs.states[:50]]  # First 5 seconds
                obs_y = [s.y for s in obs.states[:50]]
                x_min = min(x_min, min(obs_x) - 5)
                x_max = max(x_max, max(obs_x) + 5)
                y_min = min(y_min, min(obs_y) - 5)
                y_max = max(y_max, max(obs_y) + 5)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    
    scenario_path = os.path.join(RESULT_DIR, f"scenario_{output_prefix}.png")
    fig1.savefig(scenario_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {scenario_path}")
    plt.close(fig1)
    
    # Plot Frenet trajectory
    if result.success and result.frenet_states:
        fig2 = vis.plot_frenet_trajectory(
            frenet_states=result.frenet_states,
            title=f"{name} - Frenet Trajectory"
        )
        
        frenet_path = os.path.join(RESULT_DIR, f"frenet_{output_prefix}.png")
        fig2.savefig(frenet_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {frenet_path}")
        plt.close(fig2)
    
    return ScenarioResult(
        name=name,
        success=result.success,
        time_cost=result.time_cost,
        message=result.message,
        lateral_range=lateral_range
    )


def scenario_overtake() -> ScenarioResult:
    """Scenario 1: Lane change to overtake a slower vehicle."""
    
    lane = create_straight_lane(length=200.0, num_points=100)
    
    initial_state = State(
        time_stamp=0.0,
        position=np.array([10.0, 0.0]),
        angle=0.0,
        velocity=12.0,
        acceleration=0.0
    )
    
    # Slower vehicle ahead - in adjacent lane position
    obstacle = create_moving_obstacle(
        start_position=np.array([55.0, 2.0]),  # Adjacent lane offset
        velocity=np.array([6.0, 0.0]),
        duration=12.0
    )
    
    config = SscPlannerConfig(
        planning_horizon=6.0,
        target_velocity=12.0,
        weight_proximity=0.5
    )
    
    return run_scenario(
        name="Lane Change Overtake",
        lane=lane,
        initial_state=initial_state,
        obstacles=[obstacle],
        config=config,
        output_prefix="overtake"
    )


def scenario_static_obstacle() -> ScenarioResult:
    """Scenario 2: Avoid a static obstacle."""
    
    lane = create_straight_lane(length=200.0, num_points=100)
    
    initial_state = State(
        time_stamp=0.0,
        position=np.array([10.0, 0.0]),
        angle=0.0,
        velocity=10.0,
        acceleration=0.0
    )
    
    # Static obstacle on the road - offset to allow corridor generation
    obstacle = create_static_obstacle(
        position=np.array([55.0, 1.2]),  # Offset from center
        angle=0.0,
        duration=12.0
    )
    
    config = SscPlannerConfig(
        planning_horizon=6.0,
        target_velocity=10.0,
        weight_proximity=0.5
    )
    
    return run_scenario(
        name="Static Obstacle Avoidance",
        lane=lane,
        initial_state=initial_state,
        obstacles=[obstacle],
        config=config,
        output_prefix="static"
    )


def scenario_multi_vehicle() -> ScenarioResult:
    """Scenario 3: Navigate between multiple vehicles."""
    
    lane = create_straight_lane(length=200.0, num_points=100)
    
    initial_state = State(
        time_stamp=0.0,
        position=np.array([10.0, 0.0]),
        angle=0.0,
        velocity=10.0,
        acceleration=0.0
    )
    
    # Vehicle on the right lane
    obstacle1 = create_moving_obstacle(
        start_position=np.array([50.0, -2.0]),
        velocity=np.array([8.0, 0.0]),
        duration=12.0
    )
    
    # Vehicle on the left lane (further ahead)
    obstacle2 = create_moving_obstacle(
        start_position=np.array([75.0, 2.0]),
        velocity=np.array([6.0, 0.0]),
        duration=12.0
    )
    
    config = SscPlannerConfig(
        planning_horizon=7.0,
        target_velocity=11.0,
        weight_proximity=0.5
    )
    
    return run_scenario(
        name="Multi-vehicle Avoidance",
        lane=lane,
        initial_state=initial_state,
        obstacles=[obstacle1, obstacle2],
        config=config,
        output_prefix="multi"
    )


def scenario_crossing() -> ScenarioResult:
    """Scenario 4: Avoid an obstacle crossing the road."""
    
    lane = create_straight_lane(length=200.0, num_points=100)
    
    initial_state = State(
        time_stamp=0.0,
        position=np.array([10.0, 0.0]),
        angle=0.0,
        velocity=10.0,
        acceleration=0.0
    )
    
    # Obstacle crossing from left to right
    obstacle = create_moving_obstacle(
        start_position=np.array([65.0, 12.0]),
        velocity=np.array([0.0, -3.5]),
        duration=12.0,
        width=1.5,
        length=1.5  # Smaller obstacle (pedestrian-like)
    )
    
    config = SscPlannerConfig(
        planning_horizon=7.0,
        target_velocity=10.0,
        weight_proximity=0.5
    )
    
    return run_scenario(
        name="Crossing Obstacle Avoidance",
        lane=lane,
        initial_state=initial_state,
        obstacles=[obstacle],
        config=config,
        output_prefix="crossing"
    )


def main():
    print("=" * 60)
    print("SSC Planner - Multi-Scenario Test")
    print("=" * 60)
    
    # Ensure result directory exists
    os.makedirs(RESULT_DIR, exist_ok=True)
    print(f"Output directory: {RESULT_DIR}")
    
    # Run all scenarios
    results = []
    
    results.append(scenario_overtake())
    results.append(scenario_static_obstacle())
    results.append(scenario_multi_vehicle())
    results.append(scenario_crossing())
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Scenario':<30} {'Status':<10} {'Time (ms)':<12} {'Lateral Range'}")
    print("-" * 70)
    
    success_count = 0
    for r in results:
        status = "Success" if r.success else "Failed"
        lateral = f"[{r.lateral_range[0]:.2f}, {r.lateral_range[1]:.2f}]" if r.success else "N/A"
        print(f"{r.name:<30} {status:<10} {r.time_cost:<12.2f} {lateral}")
        if r.success:
            success_count += 1
    
    print("-" * 70)
    print(f"Total: {success_count}/{len(results)} scenarios successful")
    print(f"\nResults saved to: {RESULT_DIR}")
    print("=" * 60)
    
    return success_count == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
