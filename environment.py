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
    def __init__(self, uavs, targets, obstacles, config):
        self.uavs, self.targets, self.config = uavs, targets, config
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
        if self.config.USE_PHRRT_DURING_TRAINING and obstacles:
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
    def __init__(self, uavs, targets, graph, config):
        self.uavs, self.targets, self.graph, self.config = uavs, targets, graph, config
        self.n_uavs, self.n_targets = len(uavs), len(targets)
        self.n_actions = self.n_targets + 1
        self.state_dim = self._calculate_state_dim()
        self.current_uav_idx = 0
        self.reset()

    def _calculate_state_dim(self):
        """计算状态向量的维度"""
        uav_state = 1 + 1 + 1 + len(self.uavs[0].resources) # pos_x, pos_y, rem_dist, resources
        target_state = 1 + 1 + 1 + len(self.targets[0].resources) # pos_x, pos_y, value, rem_res
        return uav_state * self.n_uavs + target_state * self.n_targets
        
    def reset(self):
        """重置环境状态"""
        for uav in self.uavs: uav.reset()
        for target in self.targets: target.reset()
        self.current_uav_idx = 0
        return self._get_state()

    def _get_state(self):
        """构建当前环境的状态向量"""
        uav = self.uavs[self.current_uav_idx]
        uav_states = [
            uav.current_position[0], uav.current_position[1],
            uav.max_distance - uav.current_distance,
            *uav.resources
        ]
        target_states = []
        for t in self.targets:
            target_states.extend([
                t.position[0], t.position[1],
                t.value,
                *t.remaining_resources
            ])
        return np.concatenate([uav_states, target_states])

    def step(self, action):
        """执行一个动作并返回 (新状态, 奖励, 是否完成)"""
        uav = self.uavs[self.current_uav_idx]
        is_terminal = False
        
        if action < self.n_targets:
            target = self.targets[action]
            dist = self.graph.get_dist(uav.id, target.id)

            # 检查无人机是否有能力执行该任务 (资源和航程)
            if np.all(uav.resources >= target.remaining_resources) and (uav.current_distance + dist) <= uav.max_distance:
                reward = self._calculate_reward(uav, target, dist)
                uav.resources -= target.remaining_resources
                uav.current_position = target.position
                uav.current_distance += dist
                uav.task_sequence.append(target.id)
                target.allocated_uavs.append(uav.id)
                target.remaining_resources.fill(0)
            else:
                reward = -5.0 # 对无效选择的惩罚
        else: # action == self.n_targets (选择回家/结束)
            reward = 0.5 # 对完成任务序列的轻微奖励

        # 切换到下一个无人机或结束回合
        self.current_uav_idx += 1
        if self.current_uav_idx >= self.n_uavs:
            is_terminal = True
        
        next_state = self._get_state() if not is_terminal else np.zeros(self.state_dim)
        return next_state, reward, is_terminal

    def _calculate_reward(self, uav, target, dist):
        """(已修订) 融合多种因素的奖励函数"""
        # 1. 基础奖励：目标价值
        reward = target.value
        # 2. 资源贡献奖励
        res_contrib = np.sum(target.resources) / (np.sum(uav.initial_resources) + 1e-6)
        reward += res_contrib * 5
        # 3. 联盟奖励 (如果其他无人机已访问过此目标，可能意味着协同)
        if len(target.allocated_uavs) > 0: reward += 2.0
        # 4. 距离惩罚
        reward -= (dist / uav.max_distance) * 2
        # 5. 负载均衡惩罚 (鼓励无人机平均分担任务)
        total_tasks = sum(len(u.task_sequence) for u in self.uavs)
        if len(self.uavs) > 0:
            avg_tasks = total_tasks / len(self.uavs)
            reward -= abs(len(uav.task_sequence) - avg_tasks) * 0.5
        return reward