# -*- coding: utf-8 -*-
# 文件名: rllib_env.py
# 描述: 适配Ray RLlib的UAV任务分配环境，继承自gymnasium.Env

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional
import torch

from entities import UAV, Target
from path_planning import PHCurveRRTPlanner
from config import Config

class UAVTaskEnvRLlib(gym.Env):
    """
    适配Ray RLlib的UAV任务分配环境
    继承自gymnasium.Env，实现标准接口
    """
    
    def __init__(self, uavs: list, targets: list, graph, obstacles: list, config: Config):
        super().__init__()
        
        self.uavs = uavs
        self.targets = targets
        self.graph = graph
        self.obstacles = obstacles
        self.config = config
        
        # 环境参数
        self.load_balance_penalty = config.LOAD_BALANCE_PENALTY
        self.alliance_bonus = 100.0
        self.use_phrrt_in_training = config.USE_PHRRT_DURING_TRAINING
        self.invalid_action_penalty = -50.0
        self.marginal_utility_threshold = 0.8
        self.crowding_penalty_factor = 0.5
        
        # 计算状态和动作空间维度
        self.state_dim = self._calculate_state_dim()
        self.action_dim = self._calculate_action_dim()
        
        # 定义观察空间 (连续空间)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.state_dim,), 
            dtype=np.float32
        )
        
        # 定义动作空间 (离散空间)
        self.action_space = spaces.Discrete(self.action_dim)
        
        # 动作映射
        self._create_action_mapping()
        
    def _calculate_state_dim(self) -> int:
        """计算状态向量维度"""
        # 目标信息：位置(2) + 剩余资源(2) + 总资源(2) = 6
        target_state_dim = 6 * len(self.targets)
        
        # 无人机信息：位置(2) + 资源(2) + 航向(1) + 距离(1) = 6
        uav_state_dim = 6 * len(self.uavs)
        
        # 协同信息：每个目标的拥挤度(1) + 完成度(1) + 边际效用(1) + 紧迫度(1) = 4
        collaboration_dim = 4 * len(self.targets)
        
        # 全局信息：总体完成度(1) + 资源均衡度(1) = 2
        global_dim = 2
        
        return target_state_dim + uav_state_dim + collaboration_dim + global_dim
    
    def _calculate_action_dim(self) -> int:
        """计算动作空间维度"""
        # 动作格式：(target_id, uav_id, phi_idx)
        # 总动作数 = 目标数 × 无人机数 × 方向数
        n_targets = len(self.targets)
        n_uavs = len(self.uavs)
        n_phi = getattr(self.graph, 'n_phi', 6)  # 默认6个方向
        
        return n_targets * n_uavs * n_phi
    
    def _create_action_mapping(self):
        """创建动作索引到实际动作的映射"""
        self.action_to_tuple = {}
        self.tuple_to_action = {}
        
        action_idx = 0
        n_phi = getattr(self.graph, 'n_phi', 6)
        
        for target in self.targets:
            for uav in self.uavs:
                for phi_idx in range(n_phi):
                    action_tuple = (target.id, uav.id, phi_idx)
                    self.action_to_tuple[action_idx] = action_tuple
                    self.tuple_to_action[action_tuple] = action_idx
                    action_idx += 1
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """重置环境状态"""
        super().reset(seed=seed)
        
        # 重置所有实体
        for uav in self.uavs:
            uav.reset()
        for target in self.targets:
            target.reset()
        
        # 返回初始状态和信息
        initial_state = self._get_state()
        info = {
            'episode': {'r': 0.0, 'l': 0},
            'total_reward': 0.0,
            'step_count': 0
        }
        
        return initial_state, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """执行动作并返回结果"""
        # 将离散动作转换为实际动作元组
        target_id, uav_id, phi_idx = self.action_to_tuple[action]
        
        # 查找对应的目标无人机
        target = next((t for t in self.targets if t.id == target_id), None)
        uav = next((u for u in self.uavs if u.id == uav_id), None)
        
        if not target or not uav:
            return self._get_state(), -100.0, True, False, {'error': 'invalid_action'}
        
        # 计算实际贡献
        actual_contribution = np.minimum(target.remaining_resources, uav.resources)
        
        # 检查无效行动
        if np.all(actual_contribution <= 0):
            penalty = self._calculate_invalid_penalty(target, uav)
            return self._get_state(), penalty, False, False, {'invalid_action': True}
        
        # 记录目标完成前的状态
        was_satisfied = np.all(target.remaining_resources <= 0)
        
        # 计算路径长度
        path_len = self._calculate_path_length(uav, target, phi_idx)
        travel_time = path_len / uav.velocity_range[1]
        
        # 更新状态
        self._update_state(uav, target, actual_contribution, phi_idx)
        
        # 检查是否完成所有目标
        done = self._check_completion()
        
        # 计算奖励
        reward = self._calculate_reward(target, uav, actual_contribution, 
                                     was_satisfied, travel_time, done)
        
        # 构建信息字典（确保所有值都是Python原生类型）
        info = {
            'target_id': int(target_id),
            'uav_id': int(uav_id),
            'phi_idx': int(phi_idx),
            'actual_contribution': float(np.sum(actual_contribution)),
            'path_length': float(path_len),
            'travel_time': float(travel_time),
            'done': bool(done)
        }
        
        return self._get_state(), reward, done, False, info
    
    def _calculate_invalid_penalty(self, target: Target, uav: UAV) -> float:
        """计算无效行动的惩罚"""
        target_completion = 1.0 - (np.sum(target.remaining_resources) / np.sum(target.resources))
        uav_resources = np.sum(uav.resources)
        
        if target_completion >= 1.0:
            return self.invalid_action_penalty * 1.5
        elif uav_resources <= 0:
            return self.invalid_action_penalty * 1.5
        else:
            return self.invalid_action_penalty
    
    def _calculate_path_length(self, uav: UAV, target: Target, phi_idx: int) -> float:
        """计算路径长度"""
        if self.use_phrrt_in_training:
            start_heading = uav.heading if not uav.task_sequence else self.graph.phi_set[uav.task_sequence[-1][1]]
            planner = PHCurveRRTPlanner(uav.current_position, target.position, 
                                      start_heading, self.graph.phi_set[phi_idx], 
                                      self.obstacles, self.config)
            plan_result = planner.plan()
            return plan_result[1] if plan_result else np.linalg.norm(uav.current_position - target.position)
        else:
            return np.linalg.norm(uav.current_position - target.position)
    
    def _update_state(self, uav: UAV, target: Target, contribution: np.ndarray, phi_idx: int):
        """更新环境状态"""
        uav.resources = uav.resources.astype(np.float64) - contribution.astype(np.float64)
        target.remaining_resources = target.remaining_resources.astype(np.float64) - contribution.astype(np.float64)
        
        if uav.id not in {a[0] for a in target.allocated_uavs}:
            target.allocated_uavs.append((uav.id, phi_idx))
        
        uav.task_sequence.append((target.id, phi_idx))
        uav.current_position = target.position
        uav.heading = self.graph.phi_set[phi_idx]
    
    def _check_completion(self) -> bool:
        """检查是否完成所有目标"""
        total_satisfied = sum(np.all(t.remaining_resources <= 0) for t in self.targets)
        total_targets = len(self.targets)
        return bool(total_satisfied == total_targets)
    
    def _calculate_reward(self, target: Target, uav: UAV, contribution: np.ndarray,
                         was_satisfied: bool, travel_time: float, done: bool) -> float:
        """计算奖励"""
        # 1. 目标完成奖励
        now_satisfied = np.all(target.remaining_resources <= 0)
        new_satisfied = int(now_satisfied and not was_satisfied)
        target_completion_reward = 1500 if new_satisfied else 0
        
        # 2. 边际效用递减奖励
        target_initial_total = np.sum(target.resources)
        target_remaining_before = np.sum(target.remaining_resources + contribution)
        target_remaining_after = np.sum(target.remaining_resources)
        completion_ratio_before = 1.0 - (target_remaining_before / target_initial_total)
        completion_ratio_after = 1.0 - (target_remaining_after / target_initial_total)
        completion_improvement = completion_ratio_after - completion_ratio_before
        
        marginal_utility_factor = 1.0
        if completion_ratio_after > self.marginal_utility_threshold:
            excess_ratio = (completion_ratio_after - self.marginal_utility_threshold) / (1.0 - self.marginal_utility_threshold)
            marginal_utility_factor = 1.0 - (excess_ratio * self.crowding_penalty_factor)
        
        completion_progress_reward = completion_improvement * 800 * marginal_utility_factor
        
        # 3. 资源满足度奖励
        resource_satisfaction_ratio = np.sum(contribution) / np.sum(target.remaining_resources + contribution)
        resource_satisfaction_reward = resource_satisfaction_ratio * 200 * marginal_utility_factor
        
        # 4. 协作奖励
        collaboration_bonus = 0
        if len(target.allocated_uavs) > 1:
            crowding_factor = 1.0
            if len(target.allocated_uavs) > 2:
                crowding_factor = max(0.5, 1.0 - (len(target.allocated_uavs) - 2) * 0.2)
            collaboration_bonus = self.alliance_bonus * len(target.allocated_uavs) * 0.5 * crowding_factor
        
        # 5. 效率奖励
        efficiency_reward = 100 / (travel_time + 1) if travel_time > 0 else 0
        
        # 6. 路径惩罚
        path_penalty = -travel_time * 0.1
        
        # 7. 拥挤惩罚
        crowding_penalty = -(len(target.allocated_uavs) - 3) * 30 if len(target.allocated_uavs) > 3 else 0
        
        # 8. 完成奖励
        completion_bonus = 0
        if done:
            total_time = sum(len(uav.task_sequence) for uav in self.uavs)
            if total_time > 0:
                completion_bonus = 50000 / total_time
        
        # 9. 资源节约奖励
        resource_conservation_bonus = 0
        if done:
            total_remaining_resources = sum(np.sum(uav.resources) for uav in self.uavs)
            total_initial_resources = sum(np.sum(uav.initial_resources) for uav in self.uavs)
            if total_initial_resources > 0:
                conservation_ratio = total_remaining_resources / total_initial_resources
                resource_conservation_bonus = conservation_ratio * 1000
        
        # 综合奖励
        total_reward = (
            target_completion_reward +
            completion_progress_reward +
            resource_satisfaction_reward +
            collaboration_bonus +
            efficiency_reward +
            path_penalty +
            crowding_penalty +
            completion_bonus +
            resource_conservation_bonus
        )
        
        # 奖励缩放
        if hasattr(self.config, 'reward_scaling_factor') and self.config.reward_scaling_factor > 0:
            total_reward = total_reward / self.config.reward_scaling_factor
        else:
            total_reward = total_reward / 1000.0
        
        # 限制奖励范围
        min_reward = getattr(self.config, 'min_reward', -10.0)
        max_reward = getattr(self.config, 'max_reward', 10.0)
        total_reward = np.clip(total_reward, min_reward, max_reward)
        
        return total_reward
    
    def _get_state(self) -> np.ndarray:
        """构建当前环境的状态向量"""
        state = []
        
        # 目标信息：位置、剩余资源、总资源
        for t in self.targets:
            state.extend([*t.position, *t.remaining_resources, *t.resources])
        
        # 无人机信息：位置、资源、航向、距离
        for u in self.uavs:
            state.extend([*u.current_position, *u.resources, u.heading, u.current_distance])
        
        # 协同信息
        for t in self.targets:
            # 目标拥挤度
            allocated_uav_count = len(t.allocated_uavs)
            state.append(allocated_uav_count)
            
            # 目标完成度
            completion_ratio = 1.0 - (np.sum(t.remaining_resources) / np.sum(t.resources))
            state.append(completion_ratio)
            
            # 目标边际效用
            marginal_utility = 1.0
            if completion_ratio > self.marginal_utility_threshold:
                excess_ratio = (completion_ratio - self.marginal_utility_threshold) / (1.0 - self.marginal_utility_threshold)
                marginal_utility = 1.0 - (excess_ratio * self.crowding_penalty_factor)
            state.append(marginal_utility)
            
            # 目标资源需求紧迫度
            urgency_ratio = np.sum(t.remaining_resources) / np.sum(t.resources)
            state.append(urgency_ratio)
        
        # 全局协同信息
        # 总体完成度
        total_completion = sum(1.0 - (np.sum(t.remaining_resources) / np.sum(t.resources)) for t in self.targets) / len(self.targets)
        state.append(total_completion)
        
        # 资源分配均衡度
        uav_utilizations = []
        for u in self.uavs:
            initial_total = np.sum(u.initial_resources)
            current_total = np.sum(u.resources)
            if initial_total > 0:
                utilization = 1.0 - (current_total / initial_total)
                uav_utilizations.append(utilization)
        
        if uav_utilizations:
            balance_score = 1.0 - np.std(uav_utilizations)
            state.append(balance_score)
        else:
            state.append(0.0)
        
        return np.array(state, dtype=np.float32)
    
    def get_task_assignments(self) -> Dict[int, list]:
        """获取任务分配结果"""
        assignments = {uav.id: [] for uav in self.uavs}
        
        for uav in self.uavs:
            for task in uav.task_sequence:
                target_id, phi_idx = task
                assignments[uav.id].append((target_id, phi_idx))
        
        return assignments 