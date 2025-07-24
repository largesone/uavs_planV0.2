# -*- coding: utf-8 -*-
# 文件名: rllib_config.py
# 描述: RLlib专用配置文件

import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class RLlibConfig:
    """RLlib训练配置"""
    
    # 基础训练参数
    num_episodes: int = 1000
    learning_rate: float = 0.001
    gamma: float = 0.98
    batch_size: int = 128
    memory_size: int = 50000
    
    # 探索参数
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_timesteps: int = 10000
    
    # 网络参数
    fcnet_hiddens: list = None
    post_fcnet_hiddens: list = None
    
    # 并行参数
    num_rollout_workers: int = 4
    num_cpus_per_worker: int = 1
    num_gpus: int = 0
    
    # 环境参数
    scenario_name: str = "simple"
    obstacle_tolerance: float = 50.0
    n_phi: int = 6
    
    # 输出参数
    checkpoint_dir: str = "output/rllib_checkpoints"
    log_level: str = "INFO"
    
    def __post_init__(self):
        if self.fcnet_hiddens is None:
            self.fcnet_hiddens = [256, 256, 128]
        if self.post_fcnet_hiddens is None:
            self.post_fcnet_hiddens = [64]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "num_episodes": self.num_episodes,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "memory_size": self.memory_size,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_timesteps": self.epsilon_timesteps,
            "fcnet_hiddens": self.fcnet_hiddens,
            "post_fcnet_hiddens": self.post_fcnet_hiddens,
            "num_rollout_workers": self.num_rollout_workers,
            "num_cpus_per_worker": self.num_cpus_per_worker,
            "num_gpus": self.num_gpus,
            "scenario_name": self.scenario_name,
            "obstacle_tolerance": self.obstacle_tolerance,
            "n_phi": self.n_phi,
            "checkpoint_dir": self.checkpoint_dir,
            "log_level": self.log_level,
        }

def get_rllib_config(scenario_name: str = "simple") -> RLlibConfig:
    """
    获取RLlib配置
    
    Args:
        scenario_name: 场景名称
        
    Returns:
        RLlibConfig: 配置对象
    """
    config = RLlibConfig()
    config.scenario_name = scenario_name
    
    # 根据场景调整参数
    if scenario_name == "complex":
        config.num_episodes = 2000
        config.num_rollout_workers = 8
        config.fcnet_hiddens = [512, 512, 256, 128]
        config.post_fcnet_hiddens = [128, 64]
    
    return config 