# -*- coding: utf-8 -*-
# 文件名: config.py
# 描述: 集中管理项目的所有配置参数。

class Config:
    # 新增RRT路径规划参数
    RRT_ITERATIONS = 1500
    RRT_STEP_SIZE = 75.0
    """集中管理所有算法和模拟的参数"""
    # ----- 强化学习参数 -----
    RUN_TRAINING = False                        # 是否执行RL训练。若为False且有模型缓存，则直接加载。
    USE_PHRRT_DURING_TRAINING = False          # 训练时是否使用高精度PH-RRT计算距离。False会使用快速近似距离，极大提升训练速度。
    CLEAR_MODEL_CACHE_BEFORE_TRAINING = True   # 每次运行是否清空旧的模型缓存文件。
        # [修改] 将模型、图片、报告等输出文件统一存放在 output 文件夹下
    SAVED_MODEL_PATH = 'output/models/saved_model_final_efficient.pth' # 模型保存/加载路径。

        # --- 强化学习 (RL) 训练超参数 ---
    EPISODES = 800            # 训练的总轮次。
    LEARNING_RATE = 0.0005     # 优化器的学习率。
    GAMMA = 0.98               # 折扣因子，决定未来奖励的重要性。
    BATCH_SIZE = 128           # 每次从记忆库中采样的数量。
    MEMORY_SIZE = 20000        # 记忆库的最大容量。
    EPSILON_START = 1.0         # Epsilon-greedy策略的初始探索率
    EPSILON_END = 0.01          # Epsilon-greedy策略的最终探索率
    EPSILON_DECAY = 0.9995      # Epsilon的衰减率，用于探索-利用平衡。
    EPSILON_MIN = 0.1          # Epsilon的最小值。
    TARGET_UPDATE_FREQ = 10    # 目标网络更新的频率（每N轮更新一次）。
    LOAD_BALANCE_PENALTY = 0.3 # 负载均衡惩罚系数。
    PATIENCE = 30              # 早停耐心值，连续N轮奖励无提升则停止训练。
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
    RUN_TRAINING = True         # 是否执行训练过程
    SHOW_VISUALIZATION = False  # 是否显示最终结果的可视化图表
    CLEAR_MODEL_CACHE = False   # 是否在每次运行时强制重新训练 (删除已有模型)