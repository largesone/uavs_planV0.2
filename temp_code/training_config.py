# -*- coding: utf-8 -*-
# 文件名: training_config.py
# 描述: 统一管理所有训练相关参数

from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch

@dataclass
class TrainingConfig:
    """训练配置类 - 统一管理所有训练参数"""
    
    # ===== 基础训练参数 =====
    episodes: int = 1000                    # 训练轮次
    learning_rate: float = 0.001           # 学习率
    gamma: float = 0.98                    # 折扣因子
    batch_size: int = 128                  # 批次大小
    memory_size: int = 20000               # 记忆库大小
    
    # ===== 探索策略参数 =====
    epsilon_start: float = 1.0             # 初始探索率
    epsilon_end: float = 0.05              # 最终探索率
    epsilon_decay: float = 0.999           # 探索率衰减
    epsilon_min: float = 0.15              # 最小探索率
    
    # ===== 网络更新参数 =====
    target_update_freq: int = 10           # 目标网络更新频率
    patience: int = 50                     # 早停耐心值
    log_interval: int = 10                 # 日志打印间隔
    
    # ===== 自适应训练参数 =====
    use_adaptive_training: bool = True     # 是否使用自适应训练
    monitoring_window: int = 50            # 监控窗口大小
    min_episodes_before_detection: int = 100  # 最小检测轮次
    
    # ===== 早熟检测参数 =====
    stagnation_threshold: float = 0.02     # 停滞检测阈值
    oscillation_threshold: int = 8         # 振荡检测阈值
    convergence_trend_threshold: float = 0.1  # 收敛趋势阈值
    
    # ===== 干预策略参数 =====
    epsilon_adjustment_strong: float = 0.3    # 强烈干预探索率调整
    epsilon_adjustment_medium: float = 0.2    # 中等干预探索率调整
    epsilon_adjustment_light: float = 0.1     # 轻微干预探索率调整
    learning_rate_adjustment_strong: float = 1.5  # 强烈干预学习率调整
    learning_rate_adjustment_medium: float = 1.2  # 中等干预学习率调整
    
    # ===== 检查点参数 =====
    checkpoint_save_freq: int = 50         # 检查点保存频率
    max_checkpoints: int = 5               # 最大检查点数量
    
    # ===== 调试参数 =====
    verbose: bool = True                   # 详细输出
    debug_mode: bool = False               # 调试模式
    save_training_history: bool = True     # 保存训练历史
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'episodes': self.episodes,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'batch_size': self.batch_size,
            'memory_size': self.memory_size,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'target_update_freq': self.target_update_freq,
            'patience': self.patience,
            'log_interval': self.log_interval,
            'use_adaptive_training': self.use_adaptive_training,
            'monitoring_window': self.monitoring_window,
            'min_episodes_before_detection': self.min_episodes_before_detection,
            'stagnation_threshold': self.stagnation_threshold,
            'oscillation_threshold': self.oscillation_threshold,
            'convergence_trend_threshold': self.convergence_trend_threshold,
            'epsilon_adjustment_strong': self.epsilon_adjustment_strong,
            'epsilon_adjustment_medium': self.epsilon_adjustment_medium,
            'epsilon_adjustment_light': self.epsilon_adjustment_light,
            'learning_rate_adjustment_strong': self.learning_rate_adjustment_strong,
            'learning_rate_adjustment_medium': self.learning_rate_adjustment_medium,
            'checkpoint_save_freq': self.checkpoint_save_freq,
            'max_checkpoints': self.max_checkpoints,
            'verbose': self.verbose,
            'debug_mode': self.debug_mode,
            'save_training_history': self.save_training_history
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """从字典创建配置"""
        return cls(**config_dict)
    
    def update_from_config(self, config) -> None:
        """从现有配置对象更新参数"""
        if hasattr(config, 'EPISODES'):
            self.episodes = config.EPISODES
        if hasattr(config, 'LEARNING_RATE'):
            self.learning_rate = config.LEARNING_RATE
        if hasattr(config, 'GAMMA'):
            self.gamma = config.GAMMA
        if hasattr(config, 'BATCH_SIZE'):
            self.batch_size = config.BATCH_SIZE
        if hasattr(config, 'MEMORY_SIZE'):
            self.memory_size = config.MEMORY_SIZE
        if hasattr(config, 'EPSILON_START'):
            self.epsilon_start = config.EPSILON_START
        if hasattr(config, 'EPSILON_END'):
            self.epsilon_end = config.EPSILON_END
        if hasattr(config, 'EPSILON_DECAY'):
            self.epsilon_decay = config.EPSILON_DECAY
        if hasattr(config, 'EPSILON_MIN'):
            self.epsilon_min = config.EPSILON_MIN
        if hasattr(config, 'TARGET_UPDATE_FREQ'):
            self.target_update_freq = config.TARGET_UPDATE_FREQ
        if hasattr(config, 'PATIENCE'):
            self.patience = config.PATIENCE
        if hasattr(config, 'LOG_INTERVAL'):
            self.log_interval = config.LOG_INTERVAL
        if hasattr(config, 'USE_ADAPTIVE_TRAINING'):
            self.use_adaptive_training = config.USE_ADAPTIVE_TRAINING

class TrainingLogger:
    """训练日志管理器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.log_history = []
    
    def log_training_start(self, scenario_name: str, config_hash: str):
        """记录训练开始"""
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"开始训练: {scenario_name}")
            print(f"配置哈希: {config_hash}")
            print(f"训练参数:")
            print(f"  - 轮次: {self.config.episodes}")
            print(f"  - 学习率: {self.config.learning_rate}")
            print(f"  - 批次大小: {self.config.batch_size}")
            print(f"  - 自适应训练: {'启用' if self.config.use_adaptive_training else '禁用'}")
            print(f"{'='*60}")
    
    def log_episode(self, episode: int, reward: float, loss: float, epsilon: float, 
                   diagnosis: Optional[Dict] = None, intervention: Optional[Dict] = None):
        """记录单轮训练"""
        if self.config.verbose and (episode + 1) % self.config.log_interval == 0:
            log_msg = f"轮次 {episode+1:4d} | 奖励: {reward:6.2f} | 损失: {loss:6.4f} | 探索率: {epsilon:5.3f}"
            
            if diagnosis:
                log_msg += f" | 诊断: {diagnosis.get('convergence_quality', 'unknown')}"
            
            if intervention and intervention.get('intervention_level', 0) > 0:
                log_msg += f" | 干预: {intervention.get('message', '')}"
            
            print(log_msg)
        
        # 保存到历史记录
        self.log_history.append({
            'episode': episode,
            'reward': reward,
            'loss': loss,
            'epsilon': epsilon,
            'diagnosis': diagnosis,
            'intervention': intervention
        })
    
    def log_training_end(self, total_time: float, final_reward: float, 
                        early_stop_detected: bool = False):
        """记录训练结束"""
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"训练完成")
            print(f"  - 总时间: {total_time:.2f}秒")
            print(f"  - 最终奖励: {final_reward:.2f}")
            print(f"  - 早停检测: {'是' if early_stop_detected else '否'}")
            print(f"  - 总轮次: {len(self.log_history)}")
            print(f"{'='*60}")
    
    def log_intervention(self, episode: int, intervention_level: int, message: str):
        """记录干预事件"""
        if self.config.verbose:
            print(f"轮次 {episode+1}: 干预级别 {intervention_level} - {message}")
    
    def get_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        if not self.log_history:
            return {}
        
        rewards = [log['reward'] for log in self.log_history]
        losses = [log['loss'] for log in self.log_history if log['loss'] is not None]
        
        return {
            'total_episodes': len(self.log_history),
            'final_reward': rewards[-1] if rewards else 0,
            'best_reward': max(rewards) if rewards else 0,
            'avg_reward': sum(rewards) / len(rewards) if rewards else 0,
            'final_loss': losses[-1] if losses else 0,
            'avg_loss': sum(losses) / len(losses) if losses else 0,
            'intervention_count': sum(1 for log in self.log_history 
                                   if log.get('intervention') and log['intervention'].get('intervention_level', 0) > 0)
        }

def create_training_config_from_main_config(config) -> TrainingConfig:
    """从主配置创建训练配置"""
    training_config = TrainingConfig()
    training_config.update_from_config(config)
    return training_config 