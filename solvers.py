# -*- coding: utf-8 -*-
# 文件名: solvers.py
# 描述: 包含解决任务分配问题的核心算法，主要是基于强化学习的求解器。

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
import time

from environment import UAVTaskEnv

# =============================================================================
# section 4: 强化学习求解器
# =============================================================================

class ReplayBuffer:
    """经验回放池，用于存储和采样DQN的训练数据"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = args
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    """(已修订) 简单的深度Q网络，包含批归一化和Dropout"""
    def __init__(self, i_dim, h_dim, o_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(i_dim, h_dim)
        self.bn1 = nn.BatchNorm1d(h_dim)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(h_dim, h_dim // 2)
        self.bn2 = nn.BatchNorm1d(h_dim // 2)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(h_dim // 2, o_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        return self.fc3(x)

class GraphRLSolver:
    """(已修订) 基于图和深度强化学习的无人机任务分配求解器"""
    def __init__(self, uavs, targets, graph, obstacles, i_dim, h_dim, o_dim, config):
        self.uavs, self.targets, self.graph, self.config = uavs, targets, graph, config
        self.env = UAVTaskEnv(uavs, targets, graph, config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(i_dim, h_dim, o_dim).to(self.device)
        self.target_net = DQN(i_dim, h_dim, o_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)
        self.memory = ReplayBuffer(config.MEMORY_CAPACITY)
        self.epsilon = config.EPSILON_START
        
        # 添加动作映射
        self.target_id_map = {t.id: i for i, t in enumerate(self.env.targets)}
        self.uav_id_map = {u.id: i for i, u in enumerate(self.env.uavs)}

    def _action_to_index(self, a):
        """将动作转换为索引"""
        t_idx, u_idx, p_idx = self.target_id_map[a[0]], self.uav_id_map[a[1]], a[2]
        return t_idx * (len(self.env.uavs) * self.graph.n_phi) + u_idx * self.graph.n_phi + p_idx
    
    def _index_to_action(self, i):
        """将索引转换为动作"""
        n_u, n_p = len(self.env.uavs), self.graph.n_phi
        t_idx, u_idx, p_idx = i // (n_u * n_p), (i % (n_u * n_p)) // n_p, i % n_p
        return (self.env.targets[t_idx].id, self.env.uavs[u_idx].id, p_idx)
    
    def select_action(self, state):
        """使用Epsilon-Greedy策略选择动作"""
        if random.random() < self.epsilon:
            return torch.tensor([[random.randrange(self.env.n_actions)]], device=self.device, dtype=torch.long)
        with torch.no_grad():
            return self.policy_net(state).max(1)[1].view(1, 1)

    def optimize_model(self):
        """从经验回放池中采样并优化模型"""
        if len(self.memory) < self.config.BATCH_SIZE: return
        transitions = self.memory.sample(self.config.BATCH_SIZE)
        batch = tuple(zip(*transitions))
        
        state_batch = torch.cat(batch[0])
        action_batch = torch.cat(batch[1])
        reward_batch = torch.cat(batch[2])
        next_states_batch = torch.cat(batch[3])
        done_batch = torch.tensor(batch[4], device=self.device, dtype=torch.bool)

        q_values = self.policy_net(state_batch).gather(1, action_batch)
        next_q_values = torch.zeros(self.config.BATCH_SIZE, device=self.device)
        non_final_mask = ~done_batch
        non_final_next_states = next_states_batch[non_final_mask]
        
        if non_final_next_states.size(0) > 0:
            next_q_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
            
        expected_q_values = (next_q_values * self.config.GAMMA) + reward_batch
        loss = F.smooth_l1_loss(q_values, expected_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

    def train(self, episodes, patience, log_interval, model_save_path):
        """(已修订) 完整的训练循环，包含早停和模型保存"""
        start_time = time.time()
        best_reward = -np.inf
        patience_counter = 0

        for i_episode in range(1, episodes + 1):
            state = self.env.reset()
            state = torch.tensor([state], device=self.device, dtype=torch.float32)
            episode_reward = 0

            for t in count():
                action = self.select_action(state)
                next_state, reward, done = self.env.step(action.item())
                episode_reward += reward
                reward = torch.tensor([reward], device=self.device, dtype=torch.float32)
                next_state = torch.tensor([next_state], device=self.device, dtype=torch.float32)
                
                self.memory.push(state, action, reward, next_state, done)
                state = next_state
                self.optimize_model()
                if done: break
            
            self.epsilon = max(self.config.EPSILON_END, self.epsilon * self.config.EPSILON_DECAY)

            if i_episode % self.config.TARGET_UPDATE_INTERVAL == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            if i_episode % log_interval == 0:
                print(f"Episode {i_episode}/{episodes} | Avg Reward: {episode_reward:.2f} | Epsilon: {self.epsilon:.2f}")

            if episode_reward > best_reward:
                best_reward = episode_reward
                patience_counter = 0
                self.save_model(model_save_path)
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"早停触发于第 {i_episode} 回合。")
                break
        
        training_time = time.time() - start_time
        print(f"训练完成，耗时: {training_time:.2f}秒")
        return training_time

    def get_task_assignments(self):
        self.policy_net.eval()
        state = self.env.reset()
        assignments = {u.id: [] for u in self.env.uavs}
        done, step = False, 0
        
        while not done and step < len(self.env.targets) * len(self.env.uavs):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            action_idx = self.select_action(q_values)
            action = self._index_to_action(action_idx)
            target_id, uav_id, _ = action
            uav = self.env.uavs[uav_id - 1]
            target = self.env.targets[target_id - 1]

            # [新增] 如果无人机资源已耗尽，则不给它分配新任务
            if np.all(uav.resources <= 0):
                step += 1
                continue

            # [新增] 如果目标资源已满足，则跳过
            if np.all(target.remaining_resources <= 0):
                step += 1
                continue

            # [新增] 计算并记录实际资源消耗
            contribution = np.minimum(uav.resources, target.remaining_resources)
            if np.any(contribution > 0):  # 只有当有实际贡献时才分配任务
                assignments[uav_id].append((target_id, action[2]))  # 使用元组格式与main.py一致

            state, _, done = self.env.step(action) # env.step需要同步更新内部状态

            step += 1
        self.policy_net.train()
        return assignments
        
    def save_model(self, path):
        """(已修订) 保存模型，并确保目录存在"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy_net.state_dict(), path)
        # print(f"模型已保存至: {path}")

    def load_model(self, path):
        """加载模型"""
        if os.path.exists(path):
            self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.policy_net.eval() # 设置为评估模式
            self.target_net.eval()
            # print(f"模型已从 {path} 加载。")
            return True
        return False

# Helper function for train loop
from itertools import count