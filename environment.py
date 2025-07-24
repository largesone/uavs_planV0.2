# -*- coding: utf-8 -*-
# 文件名: environment.py
# 描述: 定义强化学习的环境，包括场景的有向图表示和任务环境本身。

import numpy as np
import itertools
from scipy.spatial.distance import cdist

from entities import UAV, Target
from path_planning import PHCurveRRTPlanner

# =============================================================================
# section 3: 场景建模与强化学习环境
# =============================================================================

class DirectedGraph:
    """(已修订) 使用numpy高效构建和管理任务场景的有向图"""
    def __init__(self, uavs, targets, n_phi, obstacles, config):
        self.uavs, self.targets, self.config = uavs, targets, config
        self.n_phi = n_phi
        self.n_uavs, self.n_targets = len(uavs), len(targets)
        self.uav_positions = np.array([u.position for u in uavs])
        self.target_positions = np.array([t.position for t in targets])
        
        self.nodes = uavs + targets
        self.node_positions = np.vstack([self.uav_positions, self.target_positions])
        self.node_map = {node.id: i for i, node in enumerate(self.nodes)}

        self.dist_matrix = self._calculate_distances(obstacles)
        self.adj_matrix = self._build_adjacency_matrix()
        self.phi_matrix = self._calculate_phi_matrix()

    def _calculate_distances(self, obstacles):
        """计算所有节点间的距离，可选地使用PH-RRT处理障碍物"""
        dist_matrix = cdist(self.node_positions, self.node_positions)
        if hasattr(self.config, 'USE_PHRRT_DURING_TRAINING') and self.config.USE_PHRRT_DURING_TRAINING and obstacles:
            for i, j in itertools.product(range(len(self.nodes)), repeat=2):
                if i == j: continue
                p1, p2 = self.node_positions[i], self.node_positions[j]
                planner = PHCurveRRTPlanner(p1, p2, 0, 0, obstacles, self.config)
                path_info = planner.plan()
                if path_info: dist_matrix[i, j] = path_info[1]
        return dist_matrix

    def _build_adjacency_matrix(self):
        """构建邻接矩阵，UAV可以飞到任何目标，目标之间不能互飞"""
        adj = np.zeros((len(self.nodes), len(self.nodes)))
        adj[:self.n_uavs, self.n_uavs:] = 1
        return adj

    def _calculate_phi_matrix(self):
        """(已修订) 高效计算所有节点对之间的相对方向分区(phi值)"""
        delta = self.node_positions[:, np.newaxis, :] - self.node_positions[np.newaxis, :, :]
        angles = np.arctan2(delta[..., 1], delta[..., 0])
        phi_matrix = np.floor((angles % (2 * np.pi)) / (2 * np.pi / self.config.GRAPH_N_PHI))
        return phi_matrix.astype(int)

    def get_dist(self, from_node_id, to_node_id):
        """获取两个节点间的距离"""
        return self.dist_matrix[self.node_map[from_node_id], self.node_map[to_node_id]]

class UAVTaskEnv:
    """(已修订) 无人机协同任务分配的强化学习环境"""
    def __init__(self, uavs, targets, graph, obstacles, config):
        self.uavs = uavs
        self.targets = targets
        self.graph = graph
        self.obstacles = obstacles
        self.config = config
        self.step_count = 0
        self.max_steps = len(targets) * len(uavs) * 2
        self.invalid_action_penalty = -75.0
        
        # 计算动作空间大小
        self.n_actions = len(targets) * len(uavs) * self.graph.n_phi
        
    def reset(self):
        """重置环境"""
        for uav in self.uavs:
            uav.reset()
        for target in self.targets:
            target.reset()
        self.step_count = 0
        return self._get_state()

    def _get_state(self):
        """获取当前状态"""
        state = []
        
        # 目标信息
        for target in self.targets:
            target_state = [
                target.position[0], target.position[1],
                target.resources[0], target.resources[1],
                target.value,
                target.remaining_resources[0], target.remaining_resources[1]
            ]
            state.extend(target_state)
        
        # UAV信息
        for uav in self.uavs:
            uav_state = [
            uav.current_position[0], uav.current_position[1],
                uav.heading,
                uav.resources[0], uav.resources[1],
                uav.max_distance,
                uav.velocity_range[0], uav.velocity_range[1]
            ]
            state.extend(uav_state)
        
        # 协同信息
        for target in self.targets:
            for uav in self.uavs:
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
            self.step_count,
            completion_rate,
            len([u for u in self.uavs if np.any(u.resources > 0)]),
            sum(np.sum(target.remaining_resources) for target in self.targets),
            sum(np.sum(uav.resources) for uav in self.uavs),
            completed_targets,
            total_targets,
            self.max_steps - self.step_count,
            np.mean([uav.heading for uav in self.uavs]),
            np.std([uav.heading for uav in self.uavs])
        ]
        state.extend(global_state)
        
        return np.array(state, dtype=np.float32)

    def step(self, action):
        """执行一步动作"""
        self.step_count += 1
        
        # 转换动作
        target_idx, uav_idx, phi_idx = self._action_to_assignment(action)
        target = self.targets[target_idx]
        uav = self.uavs[uav_idx]
        
        # 检查动作有效性
        if not self._is_valid_action(target, uav, phi_idx):
            return self._get_state(), self.invalid_action_penalty, False, False, {
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
        uav.heading = phi_idx * (2 * np.pi / self.graph.n_phi)
        
        # 检查是否完成所有目标
        total_satisfied = sum(np.all(t.remaining_resources <= 0) for t in self.targets)
        total_targets = len(self.targets)
        done = bool(total_satisfied == total_targets)
        
        # 计算奖励
        reward = self._calculate_reward(target, uav, actual_contribution, path_len, 
                                      was_satisfied, travel_time, done)
        
        # 检查是否超时
        truncated = self.step_count >= self.max_steps
        
        # 构建信息字典
        info = {
            'target_id': int(target_idx),
            'uav_id': int(uav_idx),
            'phi_idx': int(phi_idx),
            'actual_contribution': float(np.sum(actual_contribution)),
            'path_length': float(path_len),
            'travel_time': float(travel_time),
            'done': bool(done)
        }
        
        return self._get_state(), reward, done, truncated, info

    def _action_to_assignment(self, action):
        """将动作索引转换为任务分配"""
        n_uavs = len(self.uavs)
        n_targets = len(self.targets)
        n_phi = self.graph.n_phi
        
        target_idx = action // (n_uavs * n_phi)
        remaining = action % (n_uavs * n_phi)
        uav_idx = remaining // n_phi
        phi_idx = remaining % n_phi
        
        return target_idx, uav_idx, phi_idx
    
    def _is_valid_action(self, target, uav, phi_idx):
        """检查动作是否有效"""
        if np.all(uav.resources <= 0):
            return False
        if np.all(target.remaining_resources <= 0):
            return False
        if (uav.id, phi_idx) in target.allocated_uavs:
            return False
        return True

    def calculate_simplified_reward(self, target, uav, actual_contribution, path_len, 
                                was_satisfied, travel_time, done):
        """
        简化的奖励函数，重点关注目标资源满足和死锁避免
        
        Args:
            target: 目标对象
            uav: UAV对象
            actual_contribution: 实际资源贡献
            path_len: 路径长度
            was_satisfied: 之前是否已满足目标
            travel_time: 旅行时间
            done: 是否完成所有目标
            
        Returns:
            float: 归一化的奖励值
        """
        # 1. 任务完成奖励 (最高优先级)
        if done:
            return 10.0  # 归一化后的最高奖励
        
        # 2. 目标满足奖励
        now_satisfied = np.all(target.remaining_resources <= 0)
        new_satisfied = int(now_satisfied and not was_satisfied)
        target_completion_reward = 5.0 if new_satisfied else 0.0
        
        # 3. 资源贡献奖励 (核心奖励)
        # 计算贡献比例而不是绝对值
        target_initial_total = np.sum(target.resources)
        contribution_ratio = np.sum(actual_contribution) / target_initial_total if target_initial_total > 0 else 0
        contribution_reward = contribution_ratio * 3.0  # 最高3分
        
        # 4. 零贡献惩罚 (避免死锁)
        if np.all(actual_contribution <= 0):
            return -5.0  # 严重惩罚零贡献动作
        
        # 5. 距离惩罚 (简化版)
        # 使用相对距离而不是绝对距离
        max_distance = 1000.0  # 假设的最大距离
        distance_ratio = min(path_len / max_distance, 1.0)
        distance_penalty = -distance_ratio * 1.0  # 最多-1分
        
        # 总奖励 (归一化到[-5, 10]范围)
        total_reward = target_completion_reward + contribution_reward + distance_penalty
        
        return float(total_reward)
    
    def _calculate_reward(self, target, uav, actual_contribution, path_len, 
                         was_satisfied, travel_time, done):
        """
        完全重构的奖励函数 - 正向激励为核心，动态尺度惩罚
        
        核心设计理念:
        1. 巨大的正向奖励作为核心激励
        2. 所有成本作为正奖励的动态百分比减项
        3. 塑形奖励引导探索
        4. 移除所有硬编码的巨大惩罚值
        
        奖励结构:
        - 任务完成奖励: 100.0 (核心正向激励)
        - 资源贡献奖励: 10.0-50.0 (基于贡献比例)
        - 塑形奖励: 0.1-2.0 (接近目标、协作等)
        - 动态成本: 正奖励的3-8%作为减项
        """
        
        # ===== 第一部分: 计算所有正向奖励 =====
        positive_rewards = 0.0
        reward_components = {}
        
        # 1. 任务完成的巨大正向奖励 (核心激励)
        now_satisfied = np.all(target.remaining_resources <= 0)
        new_satisfied = now_satisfied and not was_satisfied
        
        if new_satisfied:
            task_completion_reward = 100.0  # 巨大的任务完成奖励
            positive_rewards += task_completion_reward
            reward_components['task_completion'] = task_completion_reward
        
        # 2. 资源贡献奖励 (基于实际贡献的正向激励)
        contribution_reward = 0.0
        if np.sum(actual_contribution) > 0:
            target_initial_total = np.sum(target.resources)
            if target_initial_total > 0:
                # 计算贡献比例
                contribution_ratio = np.sum(actual_contribution) / target_initial_total
                
                # 基础贡献奖励: 10-50分
                base_contribution = 10.0 + 40.0 * contribution_ratio
                
                # 边际效用奖励: 对小贡献也给予鼓励
                marginal_utility = 15.0 * np.sqrt(contribution_ratio)
                
                # 高效贡献奖励: 对大比例贡献给予额外奖励
                efficiency_bonus = 0.0
                if contribution_ratio > 0.3:
                    efficiency_bonus = 10.0 * (contribution_ratio - 0.3)
                
                contribution_reward = base_contribution + marginal_utility + efficiency_bonus
                positive_rewards += contribution_reward
                reward_components['contribution'] = contribution_reward
        
        # 3. 塑形奖励 - 引导探索和协作
        shaping_rewards = 0.0
        
        # 3.1 接近目标的塑形奖励
        approach_reward = self._calculate_approach_reward(uav, target)
        shaping_rewards += approach_reward
        reward_components['approach_shaping'] = approach_reward
        
        # 3.2 首次接触目标奖励
        if len(target.allocated_uavs) == 1 and target.allocated_uavs[0][0] == uav.id:
            first_contact_reward = 5.0
            shaping_rewards += first_contact_reward
            reward_components['first_contact'] = first_contact_reward
        
        # 3.3 协作塑形奖励
        collaboration_reward = self._calculate_collaboration_reward(target, uav)
        shaping_rewards += collaboration_reward
        reward_components['collaboration'] = collaboration_reward
        
        # 3.4 全局完成进度奖励
        global_progress_reward = self._calculate_global_progress_reward()
        shaping_rewards += global_progress_reward
        reward_components['global_progress'] = global_progress_reward
        
        positive_rewards += shaping_rewards
        
        # ===== 第二部分: 动态尺度成本计算 =====
        total_costs = 0.0
        
        # 确保有最小正向奖励基数，避免除零
        reward_base = max(positive_rewards, 1.0)
        
        # 1. 距离成本 - 正向奖励的3-5%
        distance_cost_ratio = 0.03 + 0.02 * min(1.0, path_len / 3000.0)  # 3%-5%
        distance_cost = reward_base * distance_cost_ratio
        total_costs += distance_cost
        reward_components['distance_cost'] = -distance_cost
        
        # 2. 时间成本 - 正向奖励的2-3%
        time_cost_ratio = 0.02 + 0.01 * min(1.0, travel_time / 60.0)  # 2%-3%
        time_cost = reward_base * time_cost_ratio
        total_costs += time_cost
        reward_components['time_cost'] = -time_cost
        
        # 3. 资源效率成本 - 如果贡献效率低
        efficiency_cost = 0.0
        if np.sum(actual_contribution) > 0:
            # 计算资源利用效率
            uav_capacity = np.sum(uav.resources)
            if uav_capacity > 0:
                utilization_ratio = np.sum(actual_contribution) / uav_capacity
                if utilization_ratio < 0.5:  # 利用率低于50%
                    efficiency_cost_ratio = 0.02 * (0.5 - utilization_ratio)  # 最多2%
                    efficiency_cost = reward_base * efficiency_cost_ratio
                    total_costs += efficiency_cost
                    reward_components['efficiency_cost'] = -efficiency_cost
        
        # ===== 第三部分: 特殊情况处理 =====
        
        # 零贡献的温和引导 (不再是硬编码的巨大惩罚)
        if np.sum(actual_contribution) <= 0:
            # 给予最小的基础奖励，但增加成本比例
            if positive_rewards == 0:
                positive_rewards = 0.5  # 最小基础奖励
                reward_components['base_reward'] = 0.5
            
            # 增加无效行动成本 (正向奖励的10%)
            ineffective_cost = positive_rewards * 0.1
            total_costs += ineffective_cost
            reward_components['ineffective_cost'] = -ineffective_cost
        
        # 全局任务完成的超级奖励
        if done:
            all_targets_satisfied = all(np.all(t.remaining_resources <= 0) for t in self.targets)
            if all_targets_satisfied:
                global_completion_reward = 200.0  # 超级完成奖励
                positive_rewards += global_completion_reward
                reward_components['global_completion'] = global_completion_reward
        
        # ===== 第四部分: 最终奖励计算 =====
        final_reward = positive_rewards - total_costs
        
        # 温和的奖励范围限制 (不再硬性裁剪)
        final_reward = np.clip(final_reward, -10.0, 300.0)
        
        # 记录详细的奖励组成
        reward_components['total_positive'] = positive_rewards
        reward_components['total_costs'] = total_costs
        reward_components['final_reward'] = final_reward
        reward_components['target_id'] = target.id
        reward_components['uav_id'] = uav.id
        reward_components['contribution_amount'] = float(np.sum(actual_contribution))
        reward_components['path_length'] = float(path_len)
        reward_components['travel_time'] = float(travel_time)
        reward_components['done'] = done
        
        self.last_reward_components = reward_components
        
        return float(final_reward)
    
    def _calculate_approach_reward(self, uav, target):
        """
        计算接近目标的塑形奖励
        
        核心思想: 如果无人机相比上一步更接近任何未完成的目标，给予微小正奖励
        这解决了目标过远导致的探索初期无正反馈问题
        """
        approach_reward = 0.0
        
        # 获取当前位置到目标的距离
        current_distance = np.linalg.norm(np.array(uav.current_position) - np.array(target.position))
        
        # 检查是否有历史位置记录
        if hasattr(uav, 'previous_position') and uav.previous_position is not None:
            previous_distance = np.linalg.norm(np.array(uav.previous_position) - np.array(target.position))
            
            # 如果更接近目标
            if current_distance < previous_distance:
                # 计算接近程度
                distance_improvement = previous_distance - current_distance
                max_improvement = 100.0  # 假设的最大改进距离
                
                # 基础接近奖励: 0.1-1.0
                base_approach = 0.1 + 0.9 * min(1.0, distance_improvement / max_improvement)
                
                # 距离越近，奖励越高
                proximity_bonus = 0.0
                if current_distance < 500.0:  # 在500米内
                    proximity_factor = (500.0 - current_distance) / 500.0
                    proximity_bonus = 0.5 * proximity_factor
                
                approach_reward = base_approach + proximity_bonus
        
        # 更新位置历史
        uav.previous_position = uav.current_position.copy()
        
        return approach_reward
    
    def _calculate_collaboration_reward(self, target, uav):
        """
        计算协作塑形奖励
        
        鼓励合理的协作，避免过度集中或过度分散
        """
        collaboration_reward = 0.0
        
        # 获取当前分配到该目标的UAV数量
        current_uav_count = len(target.allocated_uavs)
        
        if current_uav_count > 0:
            # 计算目标的资源需求量
            target_demand = np.sum(target.resources)
            
            # 估算理想的UAV数量 (基于资源需求)
            avg_uav_capacity = 50.0  # 假设平均UAV容量
            ideal_uav_count = max(1, min(4, int(np.ceil(target_demand / avg_uav_capacity))))
            
            # 协作效率奖励
            if current_uav_count <= ideal_uav_count:
                # 理想协作范围内
                efficiency_factor = 1.0 - abs(current_uav_count - ideal_uav_count) / ideal_uav_count
                collaboration_reward = 1.0 * efficiency_factor
            else:
                # 过度协作，递减奖励
                over_collaboration_penalty = (current_uav_count - ideal_uav_count) * 0.2
                collaboration_reward = max(0.2, 1.0 - over_collaboration_penalty)
            
            # 多样性奖励: 如果UAV来自不同起始位置
            if current_uav_count > 1:
                diversity_bonus = 0.3  # 基础多样性奖励
                collaboration_reward += diversity_bonus
        
        return collaboration_reward
    
    def _calculate_global_progress_reward(self):
        """
        计算全局进度塑形奖励
        
        基于整体任务完成进度给予奖励，鼓励系统性进展
        """
        if not self.targets:
            return 0.0
        
        # 计算全局完成率
        total_demand = sum(np.sum(target.resources) for target in self.targets)
        total_remaining = sum(np.sum(target.remaining_resources) for target in self.targets)
        
        if total_demand <= 0:
            return 0.0
        
        completion_rate = (total_demand - total_remaining) / total_demand
        
        # 基于完成率的进度奖励
        progress_reward = 0.0
        
        # 里程碑奖励
        milestones = [0.25, 0.5, 0.75, 0.9]
        milestone_rewards = [0.5, 1.0, 1.5, 2.0]
        
        for milestone, reward in zip(milestones, milestone_rewards):
            if completion_rate >= milestone:
                # 检查是否刚达到这个里程碑
                if not hasattr(self, '_milestone_reached'):
                    self._milestone_reached = set()
                
                if milestone not in self._milestone_reached:
                    self._milestone_reached.add(milestone)
                    progress_reward += reward
        
        # 连续进度奖励 (平滑的进度激励)
        smooth_progress = 0.2 * completion_rate
        progress_reward += smooth_progress
        
        return progress_reward