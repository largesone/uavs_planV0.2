# GCN-RL算法参数测试工具

本文档介绍了两个用于分析GCN-RL算法参数对求解结果影响的测试工具。这些工具可以帮助您找到最优的算法参数设置，提高无人机任务规划的质量。

## 文件说明

1. **quick_param_test.py**: 快速参数测试工具，专注于测试EPISODES和USE_PHRRT_DURING_TRAINING两个关键参数
2. **gcn_rl_param_analyzer.py**: 全面参数分析工具，测试更多参数组合并提供详细的可视化分析

## 快速参数测试 (quick_param_test.py)

### 功能

- 专注测试EPISODES(训练轮次)和USE_PHRRT_DURING_TRAINING(距离计算方式)两个关键参数
- 生成参数影响分析图表
- 提供最佳参数推荐

### 使用方法

```bash
python quick_param_test.py
```

### 输出

- 测试结果CSV文件: `output/quick_param_test/quick_param_test_results.csv`
- 训练轮次影响图: `output/quick_param_test/episodes_impact.png`
- 训练时间与求解质量对比图: `output/quick_param_test/time_quality_comparison.png`
- 控制台输出: 参数排名和最佳参数推荐

## 全面参数分析 (gcn_rl_param_analyzer.py)

### 功能

- 测试多个参数的组合，包括:
  - EPISODES: 训练总轮次
  - USE_PHRRT_DURING_TRAINING: 距离计算方式
  - GRAPH_N_PHI: 构建图时的离散化接近角度数量
  - LEARNING_RATE: 学习率
- 提供详细的参数影响分析和可视化
- 生成热力图展示参数组合效果

### 使用方法

```bash
python gcn_rl_param_analyzer.py
```

### 输出

- 测试结果CSV文件: `output/parameter_analysis/parameter_analysis_results.csv`
- 训练轮次影响图: `output/parameter_analysis/episodes_impact.png`
- 距离计算方式影响图: `output/parameter_analysis/phrrt_impact.png`
- 参数组合热力图: `output/parameter_analysis/heatmap_phrrt_True.png` 和 `heatmap_phrrt_False.png`
- 训练时间与求解质量权衡图: `output/parameter_analysis/time_quality_tradeoff.png`
- 控制台输出: 参数排名和详细测试结果

## 参数说明

### EPISODES (训练总轮次)

- 描述: 强化学习训练的总轮次数
- 影响: 更多的训练轮次通常会提高模型性能，但也会增加训练时间
- 测试值: 200, 500, 800 (快速测试) 或 200, 400, 800, 1200 (全面分析)

### USE_PHRRT_DURING_TRAINING (距离计算方式)

- 描述: 训练时是否使用高精度PH-RRT计算距离
- 影响: 
  - False: 使用快速近似距离，训练速度快但可能不够精确
  - True: 使用高精度PH-RRT距离，训练速度慢但更精确
- 测试值: False, True

### GRAPH_N_PHI (仅在全面分析中)

- 描述: 构建图时，每个目标节点的离散化接近角度数量
- 影响: 更多的角度可以提供更精细的路径选择，但会增加计算复杂度
- 测试值: 4, 6, 8

### LEARNING_RATE (仅在全面分析中)

- 描述: 优化器的学习率
- 影响: 影响模型收敛速度和稳定性
- 测试值: 0.0001, 0.0005, 0.001

## 评估指标

测试结果使用以下主要指标评估参数性能:

- **total_reward_score**: 综合评分，越高越好
- **completion_rate**: 任务完成率
- **satisfied_targets_rate**: 目标满足率
- **is_deadlocked**: 是否出现死锁
- **training_time**: 训练时间
- **total_runtime**: 总运行时间

## 注意事项

1. 全面参数分析需要较长时间运行，建议先使用快速参数测试获取初步结果
2. 测试结果可能因硬件配置和随机初始化而有所不同
3. 可以根据需要修改测试文件中的参数值范围