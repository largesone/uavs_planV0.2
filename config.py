# -*- coding: utf-8 -*-
# 文件名: config.py
# 描述: 集中管理项目的所有配置参数（非训练相关）

class Config:
    """集中管理所有算法和模拟的参数（非训练相关）"""
    
    # ----- 训练系统控制参数 -----
    RUN_TRAINING = False                         # 启用训练模式，让模型学习新的奖励函数
    USE_ADAPTIVE_TRAINING = True               # 是否使用自适应训练系统
    USE_PHRRT_DURING_TRAINING = True          # 训练时是否使用高精度PH-RRT计算距离
    USE_PHRRT_DURING_PLANNING = True          # 规划时是否使用高精度PH-RRT计算距离
    CLEAR_MODEL_CACHE_BEFORE_TRAINING = True   # 支持增量训练，不清空模型缓存
    SAVED_MODEL_PATH = 'output/models/saved_model_final_efficient.pth' # 模型保存/加载路径

    # ----- 路径规划参数 -----
    RRT_ITERATIONS = 1500      # RRT算法的最大迭代次数
    RRT_STEP_SIZE = 75.0       # RRT树扩展的步长
    MAX_REFINEMENT_ATTEMPTS = 15 # PH曲线路径平滑的最大尝试次数
    OBSTACLE_TOLERANCE = 50.0  # 障碍物的安全容忍距离

    # ----- 图构建参数 -----
    GRAPH_N_PHI = 6            # 构建图时，每个目标节点的离散化接近角度数量

    # ----- 模拟与评估参数 -----
    SHOW_VISUALIZATION = False  # 是否显示最终结果的可视化图表
    CLEAR_MODEL_CACHE = True   # 强制重新训练，学习新的奖励函数
    LOAD_BALANCE_PENALTY = 0.1 # 负载均衡惩罚系数

    # ----- 训练参数（已迁移到TrainingConfig，保留用于兼容性） -----
    # 注意：这些参数现在由temp_code/training_config.py中的TrainingConfig管理
    # 以下参数仅用于向后兼容，建议使用TrainingConfig
    
    # 基础训练参数
    EPISODES = 500            # 训练轮次
    LEARNING_RATE = 0.001     # 学习率
    GAMMA = 0.98               # 折扣因子
    BATCH_SIZE = 128           # 批次大小
    MEMORY_SIZE = 20000        # 记忆库大小
    
    # 探索策略参数
    EPSILON_START = 1.0         # 初始探索率
    EPSILON_END = 0.05          # 最终探索率
    EPSILON_DECAY = 0.999       # 探索率衰减
    EPSILON_MIN = 0.15          # 最小探索率
    
    # 网络更新参数
    TARGET_UPDATE_FREQ = 10    # 目标网络更新频率
    PATIENCE = 50              # 早停耐心值
    LOG_INTERVAL = 10          # 日志打印间隔

    def get_training_config(self):
        """获取训练配置对象（推荐使用）"""
        try:
            from temp_code.training_config import create_training_config_from_main_config
            return create_training_config_from_main_config(self)
        except ImportError:
            print("警告: 无法导入TrainingConfig，使用默认配置")
            return None

    def print_config_summary(self):
        """打印配置摘要"""
        print("=" * 60)
        print("配置摘要")
        print("=" * 60)
        
        # 训练系统控制
        print("训练系统控制:")
        print(f"  - RUN_TRAINING: {self.RUN_TRAINING}")
        print(f"  - USE_ADAPTIVE_TRAINING: {self.USE_ADAPTIVE_TRAINING}")
        print(f"  - USE_PHRRT_DURING_TRAINING: {self.USE_PHRRT_DURING_TRAINING}")
        print(f"  - USE_PHRRT_DURING_PLANNING: {self.USE_PHRRT_DURING_PLANNING}")
        print(f"  - CLEAR_MODEL_CACHE_BEFORE_TRAINING: {self.CLEAR_MODEL_CACHE_BEFORE_TRAINING}")
        
        # 路径规划参数
        print("\n路径规划参数:")
        print(f"  - RRT_ITERATIONS: {self.RRT_ITERATIONS}")
        print(f"  - RRT_STEP_SIZE: {self.RRT_STEP_SIZE}")
        print(f"  - MAX_REFINEMENT_ATTEMPTS: {self.MAX_REFINEMENT_ATTEMPTS}")
        print(f"  - OBSTACLE_TOLERANCE: {self.OBSTACLE_TOLERANCE}")
        
        # 图构建参数
        print("\n图构建参数:")
        print(f"  - GRAPH_N_PHI: {self.GRAPH_N_PHI}")
        
        # 模拟与评估参数
        print("\n模拟与评估参数:")
        print(f"  - SHOW_VISUALIZATION: {self.SHOW_VISUALIZATION}")
        print(f"  - CLEAR_MODEL_CACHE: {self.CLEAR_MODEL_CACHE}")
        print(f"  - LOAD_BALANCE_PENALTY: {self.LOAD_BALANCE_PENALTY}")
        
        # 训练参数（兼容性）
        print("\n训练参数（兼容性，建议使用TrainingConfig）:")
        print(f"  - EPISODES: {self.EPISODES}")
        print(f"  - LEARNING_RATE: {self.LEARNING_RATE}")
        print(f"  - GAMMA: {self.GAMMA}")
        print(f"  - BATCH_SIZE: {self.BATCH_SIZE}")
        print(f"  - EPSILON_START: {self.EPSILON_START}")
        print(f"  - EPSILON_END: {self.EPSILON_END}")
        print(f"  - PATIENCE: {self.PATIENCE}")
        print(f"  - LOG_INTERVAL: {self.LOG_INTERVAL}")
        
        print("=" * 60)
        print("注意: 训练相关参数建议使用temp_code/training_config.py中的TrainingConfig")
        print("=" * 60)

# =========================
# 配置参数说明
# =========================
# UAV_NUM: 无人机数量，推荐3~20，影响任务分配复杂度
# TARGET_NUM: 目标数量，推荐5~50，影响任务密度
# EPISODES: RL训练轮次，推荐200~2000，越大越充分但耗时增加
# LEARNING_RATE: 学习率，推荐1e-4~1e-2，过大易震荡，过小收敛慢
# USE_PHRRT_DURING_TRAINING: 是否训练时用高精度PH-RRT，True提升精度，False加快训练
# ...