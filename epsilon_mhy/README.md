# epsilon_mhy

Python实现的SSC（时空语义走廊）轨迹规划器，基于EPSILON项目重写。

## 功能

- **动态避障轨迹规划**：在已知动态障碍物轨迹的情况下规划安全轨迹
- **时空语义走廊（SSC）**：基于3D时空地图构建安全走廊
- **贝塞尔样条优化**：使用cvxpy求解QP问题，生成平滑轨迹
- **Frenet坐标变换**：支持全局坐标与Frenet坐标的相互转换
- **可视化工具**：提供matplotlib可视化支持

## 安装

```bash
cd epsilon_mhy
pip install -r requirements.txt
```

### 依赖

- numpy>=1.20.0
- scipy>=1.7.0
- cvxpy>=1.2.0
- matplotlib>=3.5.0

## 快速开始

```python
import numpy as np
from epsilon_mhy import State, Lane, SscPlanner, SscPlannerConfig
from epsilon_mhy.planning.ssc_planner import create_straight_lane

# 创建参考路径
lane = create_straight_lane(length=200.0)

# 创建初始状态
initial_state = State(
    time_stamp=0.0,
    position=np.array([10.0, 0.0]),
    angle=0.0,
    velocity=10.0
)

# 创建规划器
config = SscPlannerConfig(
    planning_horizon=5.0,
    target_velocity=12.0
)
planner = SscPlanner(config)

# 规划（无障碍物）
result = planner.plan(
    initial_state=initial_state,
    reference_lane=lane,
    obstacle_trajectories=[]
)

print(f"规划成功: {result.success}")
print(f"轨迹点数: {len(result.cartesian_states)}")
```

## 项目结构

```
epsilon_mhy/
├── core/           # 核心数据结构
│   ├── state.py    # State, FrenetState
│   ├── vehicle.py  # Vehicle, VehicleParam, ObstacleTrajectory
│   ├── lane.py     # Lane/参考路径
│   └── types.py    # 基础类型定义
├── planning/       # 规划模块
│   ├── ssc_planner.py  # SSC规划器主类
│   ├── ssc_map.py      # 3D时空地图
│   └── corridor.py     # 走廊生成
├── math/           # 数学工具
│   ├── frenet.py   # Frenet坐标变换
│   ├── bezier.py   # 贝塞尔样条
│   ├── spline.py   # 三次样条
│   └── qp_solver.py # QP优化器
├── utils/          # 工具函数
│   └── visualization.py # 可视化
├── examples/       # 使用示例
└── tests/          # 单元测试
```

## 示例

### 简单示例（无障碍物）

```bash
python examples/simple_example.py
```

### 动态避障示例

```bash
python examples/dynamic_obstacle.py
```

## 运行测试

```bash
python -m pytest tests/ -v
```

## 核心概念

### Frenet坐标系

Frenet坐标系是相对于参考路径定义的：
- **s**: 沿参考路径的纵向距离
- **d**: 相对于参考路径的横向偏移（左正右负）

### 时空语义走廊（SSC）

SSC地图是一个3D栅格（s, d, t），表示在时空中的可行区域：
1. 将障碍物轨迹投影到Frenet-时间空间
2. 基于初始轨迹种子点膨胀生成安全走廊
3. 在走廊约束下优化贝塞尔样条轨迹

### 轨迹优化

使用cvxpy求解二次规划问题：
- **目标**：最小化与参考轨迹的偏差
- **约束**：
  - 边界约束（初始/终止状态）
  - 走廊约束（保持在安全区域内）
  - 连续性约束（位置、速度、加速度连续）

## API参考

### SscPlanner

```python
class SscPlanner:
    def __init__(self, config: SscPlannerConfig)
    
    def plan(self,
             initial_state: State,
             reference_lane: Lane,
             obstacle_trajectories: List[ObstacleTrajectory],
             target_velocity: float = None,
             ego_param: VehicleParam = None) -> PlanningResult
```

### SscPlannerConfig

```python
@dataclass
class SscPlannerConfig:
    map_config: SscMapConfig      # 地图配置
    weight_proximity: float = 1.0  # 轨迹跟踪权重
    planning_horizon: float = 8.0  # 规划时域（秒）
    planning_dt: float = 0.1       # 时间步长
    target_velocity: float = 10.0  # 目标速度
```

### ObstacleTrajectory

```python
class ObstacleTrajectory:
    @classmethod
    def from_positions(cls, positions: List[Tuple[float, float, float, float]])
        """从(t, x, y, theta)列表创建障碍物轨迹"""
    
    @classmethod
    def from_state_list(cls, states: List[State])
        """从State列表创建障碍物轨迹"""
```

## 与原始EPSILON的区别

1. **移除决策模块（EUDM）**：专注于轨迹规划，不包含行为决策
2. **无ROS依赖**：独立Python库，可直接集成
3. **简化接口**：用户直接提供参考路径和障碍物轨迹
4. **使用cvxpy**：替代OOQP，更易安装和使用

## 许可证

本项目基于EPSILON项目，遵循相同的开源许可证。
