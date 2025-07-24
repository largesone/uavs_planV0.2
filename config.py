# -*- coding: utf-8 -*-
# 文件名: config.py
# 描述: 统一管理项目的所有配置参数，包括训练配置

import os
import pickle
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

@dataclass
class TrainingConfig:
    """训练配置类 - 统一管理所有训练参数"""
    
    # ===== 基础训练参数 =====
    episodes: int = 2000                   # 训练轮次
    learning_rate: float = 0.0001          # 提高学习率，加快收敛
    gamma: float = 0.99                    # 提高折扣因子，更重视长期奖励
    batch_size: int = 64                   # 减小批次大小，提高更新频率
    memory_size: int = 15000               # 适当减小记忆库，避免过旧经验
    
    # ===== 探索策略参数 =====
    epsilon_start: float = 0.9             # 降低初始探索率
    epsilon_end: float = 0.1               # 提高最终探索率，保持适度探索
    epsilon_decay: float = 0.9995          # 放缓探索率衰减
    epsilon_min: float = 0.1               # 提高最小探索率
    
    # ===== 网络更新参数 =====
    target_update_freq: int = 20           # 降低目标网络更新频率，增加稳定性
    patience: int = 100                    # 增加早停耐心值
    log_interval: int = 20                 # 减少日志输出频率
    
    # ===== 梯度裁剪参数 =====
    use_gradient_clipping: bool = True     # 是否使用梯度裁剪
    max_grad_norm: float = 1.0             # 最大梯度范数
    
    # ===== 优先经验回放参数 =====
    use_prioritized_replay: bool = True    # 是否使用优先经验回放
    per_alpha: float = 0.6                 # 优先级指数 (0=均匀采样, 1=完全优先级采样)
    per_beta_start: float = 0.4            # 重要性采样权重初始值
    per_beta_frames: int = 100000          # β从初始值增长到1.0的帧数
    per_epsilon: float = 1e-6              # 防止优先级为0的小值
    
    # ===== 调试参数 =====
    verbose: bool = True                   # 详细输出
    debug_mode: bool = False               # 调试模式
    save_training_history: bool = True     # 保存训练历史

class Config:
    """统一管理所有算法和模拟的参数"""
    
    def __init__(self):
        # ----- 训练系统控制参数 -----
        self.TRAINING_MODE = 'training'                 # 训练模式: 'training'(训练), 'inference'(推理)
        self.USE_PHRRT_DURING_TRAINING = True          # 训练时是否使用高精度PH-RRT计算距离
        self.USE_PHRRT_DURING_PLANNING = True           # 规划时是否使用高精度PH-RRT计算距离
        self.SAVED_MODEL_PATH = 'output/models/saved_model_final.pth' # 模型保存/加载路径
        
        # ----- 网络结构选择参数 -----
        # 网络结构类型: SimpleNetwork、DeepFCN、GAT、DeepFCNResidual
        self.NETWORK_TYPE = 'DeepFCNResidual'  # 修正网络类型名称
        
        # ----- 路径规划参数 =====
        self.RRT_ITERATIONS = 1000          # RRT迭代次数
        self.RRT_STEP_SIZE = 50.0           # RRT步长
        self.RRT_GOAL_BIAS = 0.1            # 目标偏向概率
        self.RRT_ADAPTIVE_STEP = True       # 是否使用自适应步长
        self.RRT_OBSTACLE_AWARE = True      # 是否使用障碍物感知采样
        self.RRT_MAX_ATTEMPTS = 3           # 最大尝试次数
        
        # ===== PH曲线平滑参数 =====
        self.MAX_REFINEMENT_ATTEMPTS = 5    # 最大细化尝试次数
        self.BEZIER_SAMPLES = 50            # 贝塞尔曲线采样点数
        self.OBSTACLE_TOLERANCE = 50.0      # 障碍物的安全容忍距离

        # ----- 图构建参数 -----
        self.GRAPH_N_PHI = 6                # 构建图时，每个目标节点的离散化接近角度数量

        # ----- 模拟与评估参数 -----
        self.SHOW_VISUALIZATION = False     # 是否显示最终结果的可视化图表
        self.LOAD_BALANCE_PENALTY = 0.1    # 负载均衡惩罚系数

        # ----- 奖励函数参数 -----
        self.TARGET_COMPLETION_REWARD = 1500    # 目标完成奖励
        self.MARGINAL_UTILITY_FACTOR = 1000    # 边际效用因子
        self.EFFICIENCY_REWARD_FACTOR = 500     # 效率奖励因子
        self.DISTANCE_PENALTY_FACTOR = 0.1     # 距离惩罚因子
        self.TIME_PENALTY_FACTOR = 10          # 时间惩罚因子
        self.COMPLETION_REWARD = 1000          # 完成奖励
        self.INVALID_ACTION_PENALTY = -100     # 无效动作惩罚
        self.ZERO_CONTRIBUTION_PENALTY = -50   # 零贡献惩罚
        self.DEADLOCK_PENALTY = -200           # 死锁惩罚
        self.COLLABORATION_BONUS = 200         # 协作奖励

        # ----- 训练配置对象 -----
        self.training_config = TrainingConfig()
        
        # 设置统一的训练参数访问接口
        self._setup_unified_training_params()
    
    def _setup_unified_training_params(self):
        """
        设置统一的训练参数访问接口
        所有训练相关参数都通过training_config统一管理，避免重复定义
        """
        # 为了向后兼容，提供属性访问接口
        pass
    
    # ===== 统一的训练参数访问属性 =====
    @property
    def EPISODES(self):
        return self.training_config.episodes
    
    @EPISODES.setter
    def EPISODES(self, value):
        self.training_config.episodes = value
    
    @property
    def LEARNING_RATE(self):
        return self.training_config.learning_rate
    
    @LEARNING_RATE.setter
    def LEARNING_RATE(self, value):
        self.training_config.learning_rate = value
    
    @property
    def GAMMA(self):
        return self.training_config.gamma
    
    @GAMMA.setter
    def GAMMA(self, value):
        self.training_config.gamma = value
    
    @property
    def BATCH_SIZE(self):
        return self.training_config.batch_size
    
    @BATCH_SIZE.setter
    def BATCH_SIZE(self, value):
        self.training_config.batch_size = value
    
    @property
    def MEMORY_SIZE(self):
        return self.training_config.memory_size
    
    @MEMORY_SIZE.setter
    def MEMORY_SIZE(self, value):
        self.training_config.memory_size = value
    
    @property
    def MEMORY_CAPACITY(self):
        return self.training_config.memory_size
    
    @MEMORY_CAPACITY.setter
    def MEMORY_CAPACITY(self, value):
        self.training_config.memory_size = value
    
    @property
    def EPSILON_START(self):
        return self.training_config.epsilon_start
    
    @EPSILON_START.setter
    def EPSILON_START(self, value):
        self.training_config.epsilon_start = value
    
    @property
    def EPSILON_END(self):
        return self.training_config.epsilon_end
    
    @EPSILON_END.setter
    def EPSILON_END(self, value):
        self.training_config.epsilon_end = value
    
    @property
    def EPSILON_DECAY(self):
        return self.training_config.epsilon_decay
    
    @EPSILON_DECAY.setter
    def EPSILON_DECAY(self, value):
        self.training_config.epsilon_decay = value
    
    @property
    def EPSILON_MIN(self):
        return self.training_config.epsilon_min
    
    @EPSILON_MIN.setter
    def EPSILON_MIN(self, value):
        self.training_config.epsilon_min = value
    
    @property
    def TARGET_UPDATE_FREQ(self):
        return self.training_config.target_update_freq
    
    @TARGET_UPDATE_FREQ.setter
    def TARGET_UPDATE_FREQ(self, value):
        self.training_config.target_update_freq = value
    
    @property
    def PATIENCE(self):
        return self.training_config.patience
    
    @PATIENCE.setter
    def PATIENCE(self, value):
        self.training_config.patience = value
    
    @property
    def LOG_INTERVAL(self):
        return self.training_config.log_interval
    
    @LOG_INTERVAL.setter
    def LOG_INTERVAL(self, value):
        self.training_config.log_interval = value
    
    # ===== 便捷的参数修改方法 =====
    def update_training_params(self, **kwargs):
        """
        便捷的训练参数批量更新方法
        
        使用示例:
        config.update_training_params(
            episodes=1000,
            learning_rate=0.001,
            batch_size=128
        )
        """
        for key, value in kwargs.items():
            if hasattr(self.training_config, key):
                setattr(self.training_config, key, value)
                print(f"✓ 更新训练参数: {key} = {value}")
            else:
                print(f"✗ 警告: 未知的训练参数 '{key}'")
    
    def get_training_summary(self):
        """获取当前训练参数摘要"""
        summary = {
            "基础参数": {
                "episodes": self.training_config.episodes,
                "learning_rate": self.training_config.learning_rate,
                "gamma": self.training_config.gamma,
                "batch_size": self.training_config.batch_size,
                "memory_size": self.training_config.memory_size,
            },
            "探索策略": {
                "epsilon_start": self.training_config.epsilon_start,
                "epsilon_end": self.training_config.epsilon_end,
                "epsilon_decay": self.training_config.epsilon_decay,
                "epsilon_min": self.training_config.epsilon_min,
            },
            "网络更新": {
                "target_update_freq": self.training_config.target_update_freq,
                "patience": self.training_config.patience,
                "log_interval": self.training_config.log_interval,
            },
            "优先经验回放": {
                "use_prioritized_replay": self.training_config.use_prioritized_replay,
                "per_alpha": self.training_config.per_alpha,
                "per_beta_start": self.training_config.per_beta_start,
                "per_beta_frames": self.training_config.per_beta_frames,
            }
        }
        return summary
    
    def print_training_config(self):
        """打印当前训练配置"""
        print("=" * 60)
        print("当前训练配置参数")
        print("=" * 60)
        
        summary = self.get_training_summary()
        for category, params in summary.items():
            print(f"\n{category}:")
            print("-" * 30)
            for key, value in params.items():
                print(f"  {key}: {value}")
        
        print("\n" + "=" * 60)
        
        # 新增的训练参数
        self.use_gradient_clipping = self.training_config.use_gradient_clipping
        self.max_grad_norm = self.training_config.max_grad_norm
    
    def update_training_config(self, new_config: TrainingConfig):
        """更新训练配置"""
        self.training_config = new_config
        self._setup_backward_compatibility()
    
    def get_training_config(self) -> TrainingConfig:
        """获取当前训练配置"""
        return self.training_config
    
    def load_existing_model(self, model_path: str = None) -> bool:
        """尝试加载已存在的模型"""
        if model_path is None:
            model_path = self.SAVED_MODEL_PATH
        
        if os.path.exists(model_path):
            print(f"发现已存在的模型: {model_path}")
            return True
        return False
    
    # ===== 训练模式便捷方法 =====
    def set_training_mode(self, mode: str):
        """设置训练模式"""
        valid_modes = ['training', 'inference']
        if mode not in valid_modes:
            raise ValueError(f"无效的训练模式: {mode}。有效模式: {valid_modes}")
        self.TRAINING_MODE = mode
    
    def is_training_mode(self) -> bool:
        """检查是否为训练模式"""
        return self.TRAINING_MODE == 'training'
    
    def is_inference_mode(self) -> bool:
        """检查是否为推理模式"""
        return self.TRAINING_MODE == 'inference'
    
    # 向后兼容的方法
    @property
    def RUN_TRAINING(self) -> bool:
        """向后兼容的RUN_TRAINING属性"""
        return self.is_training_mode()
    
    @RUN_TRAINING.setter
    def RUN_TRAINING(self, value: bool):
        """向后兼容的RUN_TRAINING设置器"""
        self.TRAINING_MODE = 'training' if value else 'inference'