#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
支持不同网络结构的RLlib环境
集成GAT网络和深度残差网络，用于对比不同的结构影响
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn

from entities import UAV, Target
from scenarios import get_strategic_trap_scenario
from main import DirectedGraph, calculate_economic_sync_speeds
from config import Config
from rllib_networks import create_network, get_network_info

class UAVTaskEnvRLlibWithNetworks(gym.Env):
    """支持不同网络结构的UAV任务环境 - RAY库版本"""
    
    def __init__(self, uavs: List[UAV], targets: List[Target], graph: DirectedGraph, 
                 obstacles: List, config: Config, network_type: str = 'SimpleFCN'):
        super(UAVTaskEnvRLlibWithNetworks, self).__init__()
        
        self.uavs = uavs
        self.targets = targets
        self.graph = graph
        self.obstacles = obstacles
        self.config = config
        self.network_type = network_type
        
        # 计算状态和动作空间维度
        self._calculate_state_dim()
        self.n_actions = len(uavs) * len(targets) * self.graph.n_phi
        
        # 设置观察和动作空间
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n_actions)
        
        # 初始化网络（用于对比）
        self._init_network()
        
        # 环境状态
        self.step_count = 0
        self.max_steps = len(targets) * len(uavs) * 2  # 最大步数
        self.invalid_action_penalty = -75.0
        
        print(f"UAVTaskEnvRLlibWithNetworks 初始化完成")
        print(f"  网络类型: {network_type}")
        print(f"  状态维度: {self.state_dim}")
        print(f"  动作空间: {self.n_actions}")
        print(f"  无人机数量: {len(uavs)}")
        print(f"  目标数量: {len(targets)}")
    
    def _calculate_state_dim(self):
        """计算状态维度"""
        # 目标信息: 位置(2) + 资源需求(2) + 价值(1) + 剩余资源(2) = 7
        target_info_dim = 7 * len(self.targets)
        
        # UAV信息: 位置(2) + 航向(1) + 资源(2) + 最大距离(1) + 速度范围(2) = 8
        uav_info_dim = 8 * len(self.uavs)
        
        # 协同信息: 当前任务分配状态
        coordination_dim = len(self.targets) * len(self.uavs)
        
        # 全局状态信息
        global_state_dim = 10  # 步数、完成率等
        
        self.state_dim = target_info_dim + uav_info_dim + coordination_dim + global_state_dim
    
    def _init_network(self):
        """初始化网络（用于对比）"""
        try:
            # 创建网络实例用于参数统计
            hidden_dims = [256, 128, 64]
            self.network = create_network(
                self.network_type, 
                self.state_dim, 
                hidden_dims, 
                self.n_actions
            )
            
            # 获取网络信息
            network_info = get_network_info(self.network)
            print(f"  网络参数: {network_info['total_parameters']:,}")
            print(f"  可训练参数: {network_info['trainable_parameters']:,}")
            
        except Exception as e:
            print(f"  网络初始化失败: {e}")
            self.network = None
    
    def _get_state(self) -> np.ndarray:
        """获取当前状态"""
        state = []
        
        # 目标信息
        for target in self.targets:
            target_state = [
                target.position[0], target.position[1],  # 位置
                target.resources[0], target.resources[1],  # 资源需求
                target.value,  # 价值
                target.remaining_resources[0], target.remaining_resources[1]  # 剩余资源
            ]
            state.extend(target_state)
        
        # UAV信息
        for uav in self.uavs:
            uav_state = [
                uav.current_position[0], uav.current_position[1],  # 位置
                uav.heading,  # 航向
                uav.resources[0], uav.resources[1],  # 资源
                uav.max_distance,  # 最大距离
                uav.velocity_range[0], uav.velocity_range[1]  # 速度范围
            ]
            state.extend(uav_state)
        
        # 协同信息 - 任务分配状态
        for target in self.targets:
            for uav in self.uavs:
                # 检查是否已分配
                is_assigned = any(
                    (uav.id, phi_idx) in target.allocated_uavs 
                    for phi_idx in range(self.graph.n_phi)
                )
                state.append(1.0 if is_assigned else 0.0)
        
        # 全局状态信息
        total_targets = len(self.targets)
        completed_targets = sum(
            1 for target in self.targets 
            if np.all(target.remaining_resources <= 0)
        )
        completion_rate = completed_targets / total_targets if total_targets > 0 else 0.0
        
        global_state = [
            self.step_count,  # 当前步数
            completion_rate,  # 完成率
            len([u for u in self.uavs if np.any(u.resources > 0)]),  # 可用UAV数量
            sum(np.sum(target.remaining_resources) for target in self.targets),  # 总剩余需求
            sum(np.sum(uav.resources) for uav in self.uavs),  # 总可用资源
            completed_targets,  # 已完成目标数
            total_targets,  # 总目标数
            self.max_steps - self.step_count,  # 剩余步数
            np.mean([uav.heading for uav in self.uavs]),  # 平均航向
            np.std([uav.heading for uav in self.uavs])  # 航向标准差
        ]
        state.extend(global_state)
        
        return np.array(state, dtype=np.float32)
    
    def _action_to_assignment(self, action: int) -> Tuple[int, int, int]:
        """将动作索引转换为任务分配"""
        n_uavs = len(self.uavs)
        n_targets = len(self.targets)
        n_phi = self.graph.n_phi
        
        target_idx = action // (n_uavs * n_phi)
        remaining = action % (n_uavs * n_phi)
        uav_idx = remaining // n_phi
        phi_idx = remaining % n_phi
        
        return target_idx, uav_idx, phi_idx
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """重置环境状态"""
        super().reset(seed=seed)
        
        # 重置所有实体
        for uav in self.uavs:
            uav.reset()
        for target in self.targets:
            target.reset()
        
        # 重置环境状态
        self.step_count = 0
        
        # 返回初始状态和信息
        initial_state = self._get_state()
        info = {
            'episode': {'r': 0.0, 'l': 0},
            'total_reward': 0.0,
            'step_count': 0,
            'network_type': self.network_type
        }
        
        return initial_state, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """执行一步动作"""
        self.step_count += 1
        
        # 转换动作
        target_idx, uav_idx, phi_idx = self._action_to_assignment(action)
        target = self.targets[target_idx]
        uav = self.uavs[uav_idx]
        
        # 检查动作有效性
        if not self._is_valid_action(target, uav, phi_idx):
            penalty = self.invalid_action_penalty
            return self._get_state(), penalty, False, False, {
                'invalid_action': True,
                'reason': 'invalid_assignment'
            }
        
        # 计算实际贡献
        actual_contribution = np.minimum(uav.resources, target.remaining_resources)
        
        # 检查是否有实际贡献
        if np.all(actual_contribution <= 0):
            return self._get_state(), self.invalid_action_penalty, False, False, {
                'invalid_action': True,
                'reason': 'no_contribution'
            }
        
        # 记录目标完成前的状态
        was_satisfied = np.all(target.remaining_resources <= 0)
        
        # 计算路径长度
        path_len = np.linalg.norm(uav.current_position - target.position)
        travel_time = path_len / uav.velocity_range[1]
        
        # 更新状态
        uav.resources = uav.resources.astype(np.float64) - actual_contribution.astype(np.float64)
        target.remaining_resources = target.remaining_resources.astype(np.float64) - actual_contribution.astype(np.float64)
        
        if uav_idx not in {a[0] for a in target.allocated_uavs}:
            target.allocated_uavs.append((uav_idx, phi_idx))
        uav.task_sequence.append((target_idx, phi_idx))
        uav.current_position = target.position
        uav.heading = self.graph.phi_set[phi_idx]
        
        # 检查是否完成所有目标
        total_satisfied = sum(np.all(t.remaining_resources <= 0) for t in self.targets)
        total_targets = len(self.targets)
        done = bool(total_satisfied == total_targets)
        
        # 计算奖励
        reward = self._calculate_reward(target, uav, actual_contribution, path_len, 
                                      was_satisfied, travel_time, done)
        
        # 检查是否超时
        truncated = self.step_count >= self.max_steps
        
        # 构建信息字典（确保所有值都是Python原生类型）
        info = {
            'target_id': int(target_idx),
            'uav_id': int(uav_idx),
            'phi_idx': int(phi_idx),
            'actual_contribution': float(np.sum(actual_contribution)),
            'path_length': float(path_len),
            'travel_time': float(travel_time),
            'done': bool(done),
            'network_type': self.network_type
        }
        
        return self._get_state(), reward, done, truncated, info
    
    def _is_valid_action(self, target: Target, uav: UAV, phi_idx: int) -> bool:
        """检查动作是否有效"""
        # 检查UAV是否有资源
        if np.all(uav.resources <= 0):
            return False
        
        # 检查目标是否还需要资源
        if np.all(target.remaining_resources <= 0):
            return False
        
        # 检查是否已经分配过
        if (uav.id, phi_idx) in target.allocated_uavs:
            return False
        
        return True
    
    def _calculate_reward(self, target: Target, uav: UAV, actual_contribution: np.ndarray,
                         path_len: float, was_satisfied: bool, travel_time: float, done: bool) -> float:
        """计算奖励"""
        # 1. 目标完成奖励
        now_satisfied = np.all(target.remaining_resources <= 0)
        new_satisfied = int(now_satisfied and not was_satisfied)
        target_completion_reward = 1500 if new_satisfied else 0
        
        # 2. 边际效用递减奖励
        target_initial_total = np.sum(target.resources)
        target_remaining_before = np.sum(target.remaining_resources + actual_contribution)
        target_remaining_after = np.sum(target.remaining_resources)
        completion_ratio_before = 1.0 - (target_remaining_before / target_initial_total)
        completion_ratio_after = 1.0 - (target_remaining_after / target_initial_total)
        completion_improvement = completion_ratio_after - completion_ratio_before
        
        # 边际效用递减
        marginal_utility = completion_improvement * (1.0 - completion_ratio_before)
        marginal_reward = marginal_utility * 1000
        
        # 3. 资源效率奖励
        resource_efficiency = np.sum(actual_contribution) / np.sum(uav.resources + actual_contribution)
        efficiency_reward = resource_efficiency * 500
        
        # 4. 距离惩罚
        distance_penalty = -path_len * 0.1
        
        # 5. 时间惩罚
        time_penalty = -travel_time * 10
        
        # 6. 完成奖励
        completion_reward = 1000 if done else 0
        
        # 总奖励
        total_reward = (target_completion_reward + marginal_reward + efficiency_reward + 
                       distance_penalty + time_penalty + completion_reward)
        
        return float(total_reward)


def create_uav_env_with_networks(network_type: str = 'SimpleFCN'):
    """创建指定网络类型的UAV环境"""
    def env_creator(config):
        uavs, targets, obstacles = get_strategic_trap_scenario(50.0)
        graph = DirectedGraph(uavs, targets, 6, obstacles)
        config_obj = Config()
        return UAVTaskEnvRLlibWithNetworks(uavs, targets, graph, obstacles, config_obj, network_type)
    
    return env_creator


def test_network_comparison():
    """测试不同网络结构的对比"""
    print("🧪 测试不同网络结构的对比...")
    
    network_types = ['SimpleFCN', 'DeepResidual', 'GAT']
    
    for network_type in network_types:
        print(f"\n--- 测试网络类型: {network_type} ---")
        
        try:
            # 创建环境
            env = create_uav_env_with_networks(network_type)()
            
            # 测试重置
            obs, info = env.reset()
            print(f"  状态形状: {obs.shape}")
            print(f"  网络类型: {info['network_type']}")
            
            # 测试几步
            for step in range(3):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                print(f"  步骤 {step+1}: 奖励={reward:.2f}, 终止={terminated}, 截断={truncated}")
                
                if terminated or truncated:
                    break
            
            print(f"✅ {network_type} 网络测试通过")
            
        except Exception as e:
            print(f"❌ {network_type} 网络测试失败: {e}")
    
    print("\n🎉 网络结构对比测试完成!")


if __name__ == "__main__":
    test_network_comparison() 