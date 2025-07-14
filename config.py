# -*- coding: utf-8 -*-
# 文件名: config.py
# 描述: 集中管理项目的所有配置参数。

class Config:
    # 新增RRT路径规划参数
    RRT_ITERATIONS = 1500
    RRT_STEP_SIZE = 75.0
    """集中管理所有算法和模拟的参数"""
    # ----- 强化学习参数 -----
    RUN_TRAINING = True                         # 启用训练模式，让模型学习新的奖励函数
    USE_PHRRT_DURING_TRAINING = False          # 训练时是否使用高精度PH-RRT计算距离。False会使用快速近似距离，极大提升训练速度。
    CLEAR_MODEL_CACHE_BEFORE_TRAINING = False   # 支持增量训练，不清空模型缓存
        # [修改] 将模型、图片、报告等输出文件统一存放在 output 文件夹下
    SAVED_MODEL_PATH = 'output/models/saved_model_final_efficient.pth' # 模型保存/加载路径。

        # --- 强化学习 (RL) 训练超参数 ---
    EPISODES = 1000            # 增加训练轮次
    LEARNING_RATE = 0.001     # 提高学习率，帮助跳出局部最优
    GAMMA = 0.98               # 调整折扣因子，更关注长期奖励
    BATCH_SIZE = 128           # 每次从记忆库中采样的数量。
    MEMORY_SIZE = 20000        # 记忆库的最大容量。
    EPSILON_START = 1.0         # Epsilon-greedy策略的初始探索率
    EPSILON_END = 0.05          # 提高最终探索率，增加随机探索
    EPSILON_DECAY = 0.999       # 降低衰减率，保持更长时间的探索
    EPSILON_MIN = 0.15          # 提高最小探索率
    TARGET_UPDATE_FREQ = 10    # 目标网络更新的频率（每N轮更新一次）。
    LOAD_BALANCE_PENALTY = 0.1 # 降低负载均衡惩罚系数，更关注资源满足
    PATIENCE = 50              # 增加早停耐心值，给模型更多时间学习
    LOG_INTERVAL = 10          # 每N轮打印一次训练日志。
        # --- 路径规划 (Path Planning) 参数 ---
    RRT_ITERATIONS = 1500      # RRT算法的最大迭代次数。
    RRT_STEP_SIZE = 75.0       # RRT树扩展的步长。
    MAX_REFINEMENT_ATTEMPTS = 15 # PH曲线路径平滑的最大尝试次数。
    OBSTACLE_TOLERANCE = 50.0  # 障碍物的安全容忍距离。

        # --- 图构建参数 ---
    GRAPH_N_PHI = 6            # 构建图时，每个目标节点的离散化接近角度数量。

    # ----- 模拟与评估 -----
    LOG_INTERVAL = 100          # 训练过程中的日志打印频率 (回合)
    SHOW_VISUALIZATION = False  # 是否显示最终结果的可视化图表
    CLEAR_MODEL_CACHE = True   # 强制重新训练，学习新的奖励函数