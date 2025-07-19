# -*- coding: utf-8 -*-
# 文件名: main.py
# 描述: 多无人机协同任务分配与路径规划的最终集成算法。
#      包含了从环境定义、强化学习求解、路径规划到结果可视化的完整流程。

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
from collections import deque, defaultdict
import os
import time
import pickle
import random
from typing import List, Dict, Tuple, Union, Optional
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import sys

# --- 本地模块导入 ---
from entities import UAV, Target
from path_planning import PHCurveRRTPlanner
from scenarios import get_small_scenario, get_complex_scenario, get_new_experimental_scenario, get_complex_scenario_v4, get_strategic_trap_scenario
from config import Config
from evaluate import evaluate_plan

# 允许多个OpenMP库共存，解决某些环境下的冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 设置控制台输出编码
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

# 初始化配置类
config = Config()

# =============================================================================
# section 1: 全局辅助函数与字体设置
# =============================================================================
def set_chinese_font(preferred_fonts=None, manual_font_path=None):
    """
    (增强版) 设置matplotlib支持中文显示的字体，以避免乱码和警告。

    Args:
        preferred_fonts (list, optional): 优先尝试的字体名称列表。 Defaults to None.
        manual_font_path (str, optional): 手动指定的字体文件路径，具有最高优先级。 Defaults to None.

    Returns:
        bool: 是否成功设置字体。
    """
    if manual_font_path and os.path.exists(manual_font_path):
        try:
            font_prop = FontProperties(fname=manual_font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False
            print(f"成功加载手动指定的字体: {manual_font_path}")
            return True
        except Exception as e:
            print(f"加载手动指定字体失败: {e}")
    
    if preferred_fonts is None:
        preferred_fonts = ['Source Han Sans SC', 'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'KaiTi', 'FangSong']
    
    try:
        for font in preferred_fonts:
            if findfont(FontProperties(family=font)):
                plt.rcParams["font.family"] = font
                plt.rcParams['axes.unicode_minus'] = False
                print(f"已自动设置中文字体为: {font}")
                return True
    except Exception:
        pass
    
    print("警告: 自动或手动设置中文字体失败。图片中的中文可能显示为方框。")
    return False

# 文件名: main.py
# ... (main.py 文件其他部分无改动) ...

# =============================================================================
# section 3: 核心业务逻辑 - 实体与环境
# =============================================================================
class DirectedGraph:
    """
    (最终修复版) 构建任务场景的有向图表示。
    此版本已更新，可以在构建图时感知障碍物，为所有算法提供更真实的距离估算。
    """
    def __init__(self, uavs: List[UAV], targets: List[Target], n_phi: int, obstacles: List = []):
        """
        构造函数。
        
        Args:
            uavs (List[UAV]): 无人机列表。
            targets (List[Target]): 目标列表。
            n_phi (int): 离散化角度数量。
            obstacles (List, optional): 场景中的障碍物列表。 Defaults to [].
        """
        self.uavs = uavs
        self.targets = targets
        self.n_phi = n_phi
        self.obstacles = obstacles
        self.phi_set = [2 * np.pi * i / n_phi for i in range(n_phi)]
        self.vertices = self._create_vertices()
        all_vertices_flat = sum(self.vertices['UAVs'].values(), []) + sum(self.vertices['Targets'].values(), [])
        self.vertex_to_idx = {v: i for i, v in enumerate(all_vertices_flat)}
        self.edges = self._create_edges()
        self.adjacency_matrix = self._create_adjacency_matrix()

    def _create_vertices(self) -> Dict:
        return {'UAVs': {u.id: [(-u.id, None)] for u in self.uavs}, 'Targets': {t.id: [(t.id, p) for p in self.phi_set] for t in self.targets}}

    def _create_edges(self) -> List:
        edges = []
        uav_vs = sum(self.vertices['UAVs'].values(), [])
        target_vs = sum(self.vertices['Targets'].values(), [])
        for uav_v in uav_vs:
            for target_v in target_vs:
                edges.append((uav_v, target_v))
        for t1_v in target_vs:
            for t2_v in target_vs:
                if t1_v[0] != t2_v[0]:
                    edges.append((t1_v, t2_v))
        return edges

    def _create_adjacency_matrix(self) -> np.ndarray:
        """
        在计算距离时，增加直线碰撞检测。如果两点间直线路径穿过障碍物，则距离为无穷大。
        """
        n = len(self.vertex_to_idx)
        adj = np.full((n, n), np.inf)
        np.fill_diagonal(adj, 0)
        
        pos_cache = {**{next(iter(self.vertices['UAVs'][u.id])): u.position for u in self.uavs}, 
                     **{v: t.position for t in self.targets for v in self.vertices['Targets'][t.id]}}

        for start_v, end_v in self.edges:
            p1, p2 = pos_cache[start_v], pos_cache[end_v]
            
            has_collision = False
            if self.obstacles:
                for obs in self.obstacles:
                    if obs.check_line_segment_collision(p1, p2):
                        has_collision = True
                        break
            
            if has_collision:
                adj[self.vertex_to_idx[start_v], self.vertex_to_idx[end_v]] = np.inf
            else:
                adj[self.vertex_to_idx[start_v], self.vertex_to_idx[end_v]] = np.linalg.norm(p2 - p1)
                
        return adj



class UAVTaskEnv:
    """强化学习环境：定义状态、动作、奖励和转移逻辑"""
    def __init__(self, uavs, targets, graph, obstacles, config):
        self.uavs, self.targets, self.graph, self.obstacles, self.config = uavs, targets, graph, obstacles, config
        self.load_balance_penalty = config.LOAD_BALANCE_PENALTY
        self.alliance_bonus = 100.0  # 协作奖励大幅提升
        self.use_phrrt_in_training = config.USE_PHRRT_DURING_TRAINING
        self.reset()
    def reset(self):
        for uav in self.uavs: uav.reset()
        for target in self.targets: target.reset()
        return self._get_state()
    def _get_state(self):
        state = [];
        for t in self.targets: state.extend([*t.position, *t.remaining_resources, *t.resources])
        for u in self.uavs: state.extend([*u.current_position, *u.resources, u.heading, u.current_distance])
        return np.array(state, dtype=np.float32)
    
    def step(self, action):
        target_id, uav_id, phi_idx = action
        target = next((t for t in self.targets if t.id == target_id), None)
        uav = next((u for u in self.uavs if u.id == uav_id), None)
        
        if not target or not uav: 
            return self._get_state(), -100, True, {}
        
        actual_contribution = np.minimum(target.remaining_resources, uav.resources)
        if np.all(actual_contribution <= 0): 
            return self._get_state(), -20, False, {}
        
        # 记录目标完成前的状态
        was_satisfied = np.all(target.remaining_resources <= 0)
        
        # 计算路径长度（保留原有设计）
        if self.use_phrrt_in_training:
            start_heading = uav.heading if not uav.task_sequence else self.graph.phi_set[uav.task_sequence[-1][1]]
            planner = PHCurveRRTPlanner(uav.current_position, target.position, start_heading, self.graph.phi_set[phi_idx], self.obstacles, self.config)
            plan_result = planner.plan()
            path_len = plan_result[1] if plan_result else np.linalg.norm(uav.current_position - target.position)
        else:
            path_len = np.linalg.norm(uav.current_position - target.position)
        
        # 计算旅行时间
        travel_time = path_len / uav.velocity_range[1]
        
        # 更新状态
        uav.resources -= actual_contribution
        target.remaining_resources -= actual_contribution
        if uav_id not in {a[0] for a in target.allocated_uavs}:
            target.allocated_uavs.append((uav_id, phi_idx))
        uav.task_sequence.append((target_id, phi_idx))
        uav.current_position = target.position
        uav.heading = self.graph.phi_set[phi_idx]
        
        # 检查是否完成所有目标
        total_satisfied = sum(np.all(t.remaining_resources <= 0) for t in self.targets)
        total_targets = len(self.targets)
        done = total_satisfied == total_targets
        
        # === 分层奖励设计：优先考虑目标数量和资源满足度 ===
        
        # 1. 目标完成奖励（最高优先级）
        now_satisfied = np.all(target.remaining_resources <= 0)
        new_satisfied = int(now_satisfied and not was_satisfied)
        target_completion_reward = 0
        if new_satisfied:
            target_completion_reward = 1500  # 增加新完成目标的奖励
        
        # 2. 目标接近完成奖励（新增渐进奖励）
        target_initial_total = np.sum(target.resources)
        target_remaining_before = np.sum(target.remaining_resources + actual_contribution)
        target_remaining_after = np.sum(target.remaining_resources)
        completion_ratio_before = 1.0 - (target_remaining_before / target_initial_total)
        completion_ratio_after = 1.0 - (target_remaining_after / target_initial_total)
        completion_improvement = completion_ratio_after - completion_ratio_before
        completion_progress_reward = completion_improvement * 800  # 增加渐进奖励权重
        
        # 3. 资源满足度奖励（新增）
        resource_satisfaction_ratio = np.sum(actual_contribution) / np.sum(target.remaining_resources + actual_contribution)
        resource_satisfaction_reward = resource_satisfaction_ratio * 200
        
        # 4. 协作奖励（增强）
        collaboration_bonus = 0
        if len(target.allocated_uavs) > 1:  # 多无人机协作
            collaboration_bonus = self.alliance_bonus * len(target.allocated_uavs) * 0.5
        
        # 5. 效率奖励（新增）
        efficiency_reward = 0
        if travel_time > 0:
            efficiency_reward = 100 / (travel_time + 1)  # 时间越短奖励越高
        
        # 6. 负载均衡奖励（轻微）
        load_balance_reward = 0
        if len(target.allocated_uavs) > 1:
            uav_contributions = []
            for uav_id, _ in target.allocated_uavs:
                uav = next((u for u in self.uavs if u.id == uav_id), None)
                if uav:
                    contribution = np.sum(np.minimum(target.resources, uav.initial_resources))
                    uav_contributions.append(contribution)
            if uav_contributions:
                std_contribution = np.std(uav_contributions)
                load_balance_reward = max(0, int(50 - std_contribution))  # 贡献越均衡奖励越高
        
        # 7. 路径惩罚（轻微）
        path_penalty = -travel_time * 0.1  # 轻微惩罚长路径
        
        # 综合奖励
        total_reward = (
            target_completion_reward +      # 目标完成（最高权重）
            completion_progress_reward +    # 渐进完成
            resource_satisfaction_reward +  # 资源满足
            collaboration_bonus +           # 协作奖励
            efficiency_reward +             # 效率奖励
            load_balance_reward +           # 负载均衡
            path_penalty                    # 路径惩罚
        )
        
        return self._get_state(), total_reward, done, {}


# =============================================================================
# section 4: 强化学习求解器
# =============================================================================
class GNN(nn.Module):
    """优化的神经网络，增加网络容量以提升学习能力"""
    def __init__(self, i_dim, h_dim, o_dim):
        super(GNN, self).__init__()
        # 增加网络深度和宽度
        self.l = nn.Sequential(
            nn.Linear(i_dim, h_dim * 2),  # 第一层：输入维度 -> 2倍隐藏层
            nn.ReLU(),
            nn.Dropout(0.1),  # 添加dropout防止过拟合
            nn.Linear(h_dim * 2, h_dim * 2),  # 第二层：保持宽度
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(h_dim * 2, h_dim),  # 第三层：逐渐减少到原始隐藏层
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(h_dim, h_dim // 2),  # 第四层：进一步减少
            nn.ReLU(),
            nn.Linear(h_dim // 2, o_dim)  # 输出层
        )
        
    def forward(self, x):
        return self.l(x)

class GraphRLSolver:
    """使用深度Q网络（DQN）的强化学习求解器"""
    def __init__(self, uavs, targets, graph, obstacles, i_dim, h_dim, o_dim, config):
        self.config, self.graph = config, graph; self.env = UAVTaskEnv(uavs, targets, graph, obstacles, config); self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GNN(i_dim, h_dim, o_dim).to(self.device); self.target_model = GNN(i_dim, h_dim, o_dim).to(self.device); self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE); self.memory = deque(maxlen=config.MEMORY_SIZE); self.epsilon, self.step_count = 1.0, 0; self.train_history = defaultdict(list)
        self.target_id_map = {t.id: i for i, t in enumerate(self.env.targets)}; self.uav_id_map = {u.id: i for i, u in enumerate(self.env.uavs)}
        
        # === 自适应多维闭环早熟检测系统 ===
        self.early_stop_detected = False
        self.early_stop_counter = 0
        self.intervention_applied = False
        
        # 多维度监控指标
        self.reward_history = []
        self.loss_history = []
        self.gradient_norm_history = []
        self.epsilon_history = []
        self.exploration_rate_history = []
        
        # 自适应检测参数
        self.detection_window = 50  # 检测窗口大小
        self.min_episodes_before_detection = 100  # 开始检测的最小轮次
        self.improvement_threshold = 0.01  # 改进阈值（1%）
        self.stability_threshold = 0.05  # 稳定性阈值（5%）
        
        # 早熟干预参数
        self.intervention_epsilon_boost = 0.3  # 探索率提升
        self.intervention_lr_reduction = 0.5  # 学习率降低
        self.intervention_memory_refresh = 0.3  # 记忆库刷新比例
        self.max_interventions = 3  # 最大干预次数
        
        # 自适应调整参数
        self.adaptive_detection_enabled = True
        self.adaptive_window_adjustment = True
        self.adaptive_threshold_adjustment = True
        
        # 性能基准
        self.best_reward = -float('inf')
        self.best_loss = float('inf')
        self.reward_plateau_count = 0
        self.loss_plateau_count = 0
        
        # 训练历史记录
        self.best_reward_history = []
        self.training_metrics = {
            'episode_rewards': [],
            'episode_losses': [],
            'gradient_norms': [],
            'exploration_rates': [],
            'intervention_history': []
        }
    def _action_to_index(self, a):
        t_idx, u_idx, p_idx = self.target_id_map[a[0]], self.uav_id_map[a[1]], a[2]; return t_idx * (len(self.env.uavs) * self.graph.n_phi) + u_idx * self.graph.n_phi + p_idx
    def _index_to_action(self, i):
        n_u, n_p = len(self.env.uavs), self.graph.n_phi; t_idx, u_idx, p_idx = i // (n_u * n_p), (i % (n_u * n_p)) // n_p, i % n_p
        return (self.env.targets[t_idx].id, self.env.uavs[u_idx].id, p_idx)
    def remember(self, s, a, r, ns, d): self.memory.append((s, a, r, ns, d))
    def replay(self):
        if len(self.memory) < self.config.BATCH_SIZE: return None
        minibatch = random.sample(self.memory, self.config.BATCH_SIZE); states, actions, rewards, next_states, dones = zip(*minibatch)
        s_tensor, ns_tensor = torch.FloatTensor(np.array(states)).to(self.device), torch.FloatTensor(np.array(next_states)).to(self.device)
        a_tensor, r_tensor, d_tensor = torch.LongTensor([self._action_to_index(a) for a in actions]).to(self.device), torch.FloatTensor(rewards).to(self.device), torch.FloatTensor(dones).to(self.device)
        q_vals = self.model(s_tensor).gather(1, a_tensor.unsqueeze(1)).squeeze(1)
        with torch.no_grad(): next_q_vals = self.target_model(ns_tensor).max(1)[0]
        targets = r_tensor + self.config.GAMMA * next_q_vals * (1 - d_tensor); loss = nn.MSELoss()(q_vals, targets)
        self.optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0); self.optimizer.step()
        if self.step_count % self.config.TARGET_UPDATE_FREQ == 0: self.target_model.load_state_dict(self.model.state_dict())
        return loss.item()
    def _get_valid_action_mask(self):
        n_t, n_u, n_p = len(self.env.targets), len(self.env.uavs), self.graph.n_phi; mask = torch.zeros(n_t * n_u * n_p, dtype=torch.bool, device=self.device)
        for t in self.env.targets:
            if np.all(t.remaining_resources <= 0): continue
            t_idx = self.target_id_map[t.id]; collaborators = {a[0] for a in t.allocated_uavs}
            for u in self.env.uavs:
                if u.id in collaborators: continue
                u_idx = self.uav_id_map[u.id]
                if np.any((u.resources > 0) & (t.remaining_resources > 0)):
                    start_idx = t_idx * (n_u * n_p) + u_idx * n_p; mask[start_idx: start_idx + n_p] = True
        return mask
    def _calculate_gradient_norm(self):
        """计算当前梯度的范数"""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def _detect_early_stopping(self, episode, avg_reward, avg_loss):
        """自适应多维闭环早熟检测"""
        if episode < self.min_episodes_before_detection:
            return False, {}
        
        # 更新历史记录
        self.reward_history.append(avg_reward)
        self.loss_history.append(avg_loss)
        self.epsilon_history.append(self.epsilon)
        
        # 计算梯度范数
        if len(self.reward_history) > 1:
            grad_norm = self._calculate_gradient_norm()
            self.gradient_norm_history.append(grad_norm)
        
        # 自适应调整检测窗口
        if self.adaptive_window_adjustment and episode > 200:
            # 根据训练进度动态调整窗口大小
            progress_ratio = episode / self.config.EPISODES
            self.detection_window = max(30, min(100, int(50 * (1 + progress_ratio))))
        
        # 自适应调整阈值
        if self.adaptive_threshold_adjustment and episode > 150:
            # 根据训练进度动态调整阈值
            progress_ratio = episode / self.config.EPISODES
            self.improvement_threshold = max(0.005, min(0.02, 0.01 * (1 + progress_ratio)))
            self.stability_threshold = max(0.03, min(0.08, 0.05 * (1 + progress_ratio)))
        
        # 多维度检测指标
        detection_results = {
            'reward_plateau': False,
            'loss_plateau': False,
            'gradient_stagnation': False,
            'exploration_deficiency': False,
            'overall_stagnation': False
        }
        
        # 1. 奖励停滞检测
        if len(self.reward_history) >= self.detection_window:
            recent_rewards = self.reward_history[-self.detection_window:]
            recent_max = max(recent_rewards)
            if recent_max <= self.best_reward * (1 + self.improvement_threshold):
                detection_results['reward_plateau'] = True
                self.reward_plateau_count += 1
            else:
                self.reward_plateau_count = 0
        
        # 2. 损失停滞检测
        if len(self.loss_history) >= self.detection_window:
            recent_losses = self.loss_history[-self.detection_window:]
            recent_min = min(recent_losses)
            if recent_min >= self.best_loss * (1 - self.improvement_threshold):
                detection_results['loss_plateau'] = True
                self.loss_plateau_count += 1
            else:
                self.loss_plateau_count = 0
        
        # 3. 梯度停滞检测
        if len(self.gradient_norm_history) >= self.detection_window:
            recent_grads = self.gradient_norm_history[-self.detection_window:]
            grad_std = np.std(recent_grads)
            grad_mean = np.mean(recent_grads)
            if grad_std < grad_mean * self.stability_threshold:
                detection_results['gradient_stagnation'] = True
        
        # 4. 探索不足检测
        if len(self.epsilon_history) >= self.detection_window:
            recent_epsilon = self.epsilon_history[-self.detection_window:]
            epsilon_std = np.std(recent_epsilon)
            if epsilon_std < 0.01:  # 探索率变化很小
                detection_results['exploration_deficiency'] = True
        
        # 5. 综合停滞检测
        stagnation_count = sum(detection_results.values())
        if stagnation_count >= 2:  # 至少两个维度出现停滞
            detection_results['overall_stagnation'] = True
        
        # 判断是否需要干预
        needs_intervention = (
            detection_results['overall_stagnation'] or
            self.reward_plateau_count >= 3 or
            self.loss_plateau_count >= 3
        )
        
        return needs_intervention, detection_results
    
    def _apply_early_stopping_intervention(self, detection_results):
        """应用早熟干预措施"""
        if self.intervention_applied or len(self.training_metrics['intervention_history']) >= self.max_interventions:
            return False
        
        print(f"\n=== 检测到早熟问题，应用干预措施 ===")
        print(f"检测结果: {detection_results}")
        
        # 1. 提升探索率
        old_epsilon = self.epsilon
        self.epsilon = min(1.0, self.epsilon + self.intervention_epsilon_boost)
        print(f"探索率提升: {old_epsilon:.3f} -> {self.epsilon:.3f}")
        
        # 2. 降低学习率
        old_lr = self.optimizer.param_groups[0]['lr']
        new_lr = old_lr * self.intervention_lr_reduction
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        print(f"学习率降低: {old_lr:.6f} -> {new_lr:.6f}")
        
        # 3. 刷新记忆库（部分）
        if len(self.memory) > 0:
            refresh_size = int(len(self.memory) * self.intervention_memory_refresh)
            for _ in range(refresh_size):
                if len(self.memory) > 0:
                    self.memory.popleft()
            print(f"记忆库刷新: 移除 {refresh_size} 个旧经验")
        
        # 4. 重置停滞计数器
        self.reward_plateau_count = 0
        self.loss_plateau_count = 0
        
        # 5. 记录干预历史
        intervention_record = {
            'episode': len(self.reward_history),
            'detection_results': detection_results,
            'epsilon_change': self.epsilon - old_epsilon,
            'lr_change': new_lr - old_lr,
            'memory_refresh': refresh_size if len(self.memory) > 0 else 0
        }
        self.training_metrics['intervention_history'].append(intervention_record)
        
        self.intervention_applied = True
        return True
    def train(self, episodes, patience, log_interval, model_save_path):
        """训练强化学习模型"""
        print(f"开始训练，总轮次: {episodes}")
        
        # 训练历史记录
        episode_rewards = []
        episode_losses = []
        intervention_history = []
        
        # 早停检测相关
        best_reward = float('-inf')
        patience_counter = 0
        early_stopping_triggered = False
        
        # 自适应训练参数
        current_lr = self.optimizer.param_groups[0]['lr']
        current_epsilon = self.epsilon
        
        for episode in tqdm(range(episodes), desc="训练进度"):
            state = self.env.reset()
            episode_reward = 0
            episode_loss = 0
            steps = 0
            
            while True:
                # 获取有效动作掩码
                valid_actions = self.env._get_valid_action_mask()
                if not any(valid_actions):
                    break
                
                # 选择动作
                action = self._select_action(state, valid_actions)
                next_state, reward, done, _ = self.env.step(action)
                
                # 存储经验
                self.memory.append((state, action, reward, next_state, done))
                episode_reward += reward
                state = next_state
                steps += 1
                
                # 训练网络
                if len(self.memory) >= self.batch_size:
                    loss = self.replay()
                    if loss is not None:
                        episode_loss += loss
                
                if done:
                    break
            
            # 记录历史
            episode_rewards.append(episode_reward)
            if episode_loss > 0:
                episode_losses.append(episode_loss / steps)
            else:
                episode_losses.append(0.0)
            
            # 早停检测
            if len(episode_rewards) >= 20:
                detection_results = self._detect_early_stopping(episode, episode_rewards, episode_losses)
                
                if detection_results['early_maturity_detected']:
                    early_stopping_triggered = True
                    intervention_results = self._apply_early_stopping_intervention(detection_results)
                    
                    if intervention_results['intervention_applied']:
                        # 记录干预历史
                        intervention_history.append({
                            'episode': episode,
                            'detection_type': detection_results['detection_type'],
                            'epsilon_change': intervention_results['epsilon_change'],
                            'lr_change': intervention_results['lr_change'],
                            'intervention_type': intervention_results['intervention_type']
                        })
                        
                        # 更新参数
                        current_epsilon = intervention_results['new_epsilon']
                        current_lr = intervention_results['new_lr']
                        
                        # 应用参数变化
                        self.epsilon = current_epsilon
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = current_lr
                        
                        print(f"轮次 {episode}: 检测到{detection_results['detection_type']}，应用{intervention_results['intervention_type']}干预")
                        print(f"  探索率: {current_epsilon:.4f}, 学习率: {current_lr:.6f}")
            
            # 定期保存模型
            if (episode + 1) % 100 == 0:
                self.save_model(model_save_path)
            
            # 早停检查
            if episode_reward > best_reward:
                best_reward = episode_reward
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"早停触发，轮次 {episode}")
                break
        
        # 保存最终模型
        self.save_model(model_save_path)
        
        # 生成增强收敛图
        self._plot_enhanced_convergence(model_save_path)
        
        # 生成奖励曲线报告
        self._generate_reward_curve_report(model_save_path)
        
        # 保存训练历史
        history_data = {
            'episode_rewards': episode_rewards,
            'episode_losses': episode_losses,
            'intervention_history': intervention_history,
            'early_stopping_triggered': early_stopping_triggered,
            'final_epsilon': current_epsilon,
            'final_lr': current_lr
        }
        
        history_path = model_save_path.replace('.pth', '_history.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(history_data, f)
        
        print(f"训练完成，总轮次: {len(episode_rewards)}")
        print(f"最佳奖励: {best_reward:.2f}")
        print(f"最终探索率: {current_epsilon:.4f}")
        print(f"最终学习率: {current_lr:.6f}")
        if intervention_history:
            print(f"干预次数: {len(intervention_history)}")
        
        return episode_rewards, episode_losses

    def _plot_enhanced_convergence(self, model_save_path):
        """绘制增强的训练收敛情况图表，包含早熟检测和干预信息"""
        # 创建保存路径
        save_dir = os.path.dirname(model_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 创建图表
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('增强训练收敛分析 - 包含早熟检测与干预', fontsize=16, fontweight='bold')
        
        # 1. 奖励收敛图
        ax1 = axes[0, 0]
        rewards = self.train_history['episode_rewards']
        episodes = range(1, len(rewards) + 1)
        ax1.plot(episodes, rewards, 'b-', alpha=0.6, label='每轮奖励')
        
        # 添加移动平均线
        window_size = min(50, len(rewards))
        if window_size > 0:
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            moving_episodes = range(window_size, len(rewards) + 1)
            ax1.plot(moving_episodes, moving_avg, 'r-', linewidth=2, label=f'{window_size}轮移动平均')
        
        # 标记早熟检测点
        if hasattr(self, 'early_stop_history') and self.early_stop_history:
            for ep, detected in self.early_stop_history:
                if detected:
                    ax1.axvline(x=ep, color='red', linestyle='--', alpha=0.7, label='早熟检测' if ep == self.early_stop_history[0][0] else "")
        
        # 标记干预点
        if hasattr(self, 'intervention_history') and self.intervention_history:
            for ep, intervention_type in self.intervention_history:
                ax1.axvline(x=ep, color='orange', linestyle=':', alpha=0.7, label=f'干预({intervention_type})' if ep == self.intervention_history[0][0] else "")
        
        ax1.set_title('奖励收敛曲线')
        ax1.set_xlabel('训练轮次')
        ax1.set_ylabel('奖励值')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 损失收敛图
        ax2 = axes[0, 1]
        if 'episode_losses' in self.train_history and self.train_history['episode_losses']:
            losses = self.train_history['episode_losses']
            loss_episodes = range(1, len(losses) + 1)
            ax2.plot(loss_episodes, losses, 'g-', alpha=0.6, label='每轮损失')
            
            # 添加移动平均线
            if len(losses) >= window_size:
                moving_loss_avg = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
                moving_loss_episodes = range(window_size, len(losses) + 1)
                ax2.plot(moving_loss_episodes, moving_loss_avg, 'r-', linewidth=2, label=f'{window_size}轮移动平均')
            
            ax2.set_title('损失收敛曲线')
            ax2.set_xlabel('训练轮次')
            ax2.set_ylabel('损失值')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, '无损失数据', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('损失收敛曲线')
        
        # 3. 梯度范数变化
        ax3 = axes[1, 0]
        if 'gradient_norms' in self.train_history and self.train_history['gradient_norms']:
            grad_norms = self.train_history['gradient_norms']
            grad_episodes = range(1, len(grad_norms) + 1)
            ax3.plot(grad_episodes, grad_norms, 'purple', alpha=0.6, label='梯度范数')
            
            # 添加移动平均线
            if len(grad_norms) >= window_size:
                moving_grad_avg = np.convolve(grad_norms, np.ones(window_size)/window_size, mode='valid')
                moving_grad_episodes = range(window_size, len(grad_norms) + 1)
                ax3.plot(moving_grad_episodes, moving_grad_avg, 'r-', linewidth=2, label=f'{window_size}轮移动平均')
            
            ax3.set_title('梯度范数变化')
            ax3.set_xlabel('训练轮次')
            ax3.set_ylabel('梯度范数')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, '无梯度数据', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('梯度范数变化')
        
        # 4. 探索率变化
        ax4 = axes[1, 1]
        if 'epsilon_values' in self.train_history and self.train_history['epsilon_values']:
            epsilons = self.train_history['epsilon_values']
            epsilon_episodes = range(1, len(epsilons) + 1)
            ax4.plot(epsilon_episodes, epsilons, 'orange', alpha=0.6, label='探索率')
            ax4.set_title('探索率变化')
            ax4.set_xlabel('训练轮次')
            ax4.set_ylabel('探索率')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, '无探索率数据', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('探索率变化')
        
        # 5. 早熟检测指标
        ax5 = axes[2, 0]
        if hasattr(self, 'early_stop_metrics') and self.early_stop_metrics:
            metrics = self.early_stop_metrics
            episodes = range(1, len(metrics) + 1)
            
            # 绘制多维度指标
            for metric_name, values in metrics.items():
                if len(values) == len(episodes):
                    ax5.plot(episodes, values, alpha=0.6, label=metric_name)
            
            ax5.set_title('早熟检测指标')
            ax5.set_xlabel('训练轮次')
            ax5.set_ylabel('指标值')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, '无早熟检测数据', ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('早熟检测指标')
        
        # 6. 干预历史统计
        ax6 = axes[2, 1]
        if hasattr(self, 'intervention_history') and self.intervention_history:
            intervention_types = [intervention[1] for intervention in self.intervention_history]
            intervention_counts = {}
            for intervention_type in intervention_types:
                intervention_counts[intervention_type] = intervention_counts.get(intervention_type, 0) + 1
            
            if intervention_counts:
                types = list(intervention_counts.keys())
                counts = list(intervention_counts.values())
                ax6.bar(types, counts, color=['red', 'orange', 'yellow'])
                ax6.set_title('干预类型统计')
                ax6.set_xlabel('干预类型')
                ax6.set_ylabel('干预次数')
                ax6.tick_params(axis='x', rotation=45)
        else:
            ax6.text(0.5, 0.5, '无干预数据', ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('干预类型统计')
        
        plt.tight_layout()
        
        # 保存图片
        convergence_path = model_save_path.replace('.pth', '_enhanced_convergence.png')
        plt.savefig(convergence_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"增强收敛分析图已保存至: {convergence_path}")
        
        # 生成奖励曲线详细报告
        self._generate_reward_curve_report(model_save_path)
    
    def _generate_reward_curve_report(self, model_save_path):
        """生成奖励曲线详细报告"""
        save_dir = os.path.dirname(model_save_path)
        report_path = model_save_path.replace('.pth', '_reward_curve_report.txt')
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("强化学习训练奖励曲线分析报告")
        report_lines.append("=" * 80)
        report_lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 奖励统计
        if 'episode_rewards' in self.train_history:
            rewards = self.train_history['episode_rewards']
            report_lines.append("奖励统计:")
            report_lines.append(f"  总训练轮次: {len(rewards)}")
            report_lines.append(f"  最高奖励: {max(rewards):.2f}")
            report_lines.append(f"  最低奖励: {min(rewards):.2f}")
            report_lines.append(f"  平均奖励: {np.mean(rewards):.2f}")
            report_lines.append(f"  奖励标准差: {np.std(rewards):.2f}")
            report_lines.append(f"  最终奖励: {rewards[-1]:.2f}")
            
            # 奖励趋势分析
            if len(rewards) > 10:
                recent_rewards = rewards[-10:]
                early_rewards = rewards[:10]
                recent_avg = np.mean(recent_rewards)
                early_avg = np.mean(early_rewards)
                improvement = (recent_avg - early_avg) / abs(early_avg) * 100 if early_avg != 0 else 0
                report_lines.append(f"  奖励改进: {improvement:.2f}%")
            
            # 收敛性分析
            if len(rewards) > 50:
                last_50 = rewards[-50:]
                first_50 = rewards[:50]
                convergence_ratio = np.std(last_50) / np.std(first_50) if np.std(first_50) > 0 else 0
                report_lines.append(f"  收敛性指标: {convergence_ratio:.3f} (越小越稳定)")
        
        report_lines.append("")
        
        # 早熟检测统计
        if hasattr(self, 'early_stop_history') and self.early_stop_history:
            report_lines.append("早熟检测统计:")
            total_detections = sum(1 for _, detected in self.early_stop_history if detected)
            report_lines.append(f"  总检测次数: {len(self.early_stop_history)}")
            report_lines.append(f"  早熟检测次数: {total_detections}")
            report_lines.append(f"  早熟检测率: {total_detections/len(self.early_stop_history)*100:.1f}%")
        
        report_lines.append("")
        
        # 干预统计
        if hasattr(self, 'intervention_history') and self.intervention_history:
            report_lines.append("干预统计:")
            intervention_types = {}
            for _, intervention_type in self.intervention_history:
                intervention_types[intervention_type] = intervention_types.get(intervention_type, 0) + 1
            
            for intervention_type, count in intervention_types.items():
                report_lines.append(f"  {intervention_type}: {count}次")
        
        report_lines.append("")
        
        # 训练建议
        report_lines.append("训练建议:")
        if 'episode_rewards' in self.train_history:
            rewards = self.train_history['episode_rewards']
            if len(rewards) > 0:
                final_reward = rewards[-1]
                max_reward = max(rewards)
                
                if final_reward < max_reward * 0.8:
                    report_lines.append("  - 当前奖励远低于历史最佳，建议调整学习率或增加训练轮次")
                
                if len(rewards) > 100 and np.std(rewards[-50:]) < np.std(rewards[:50]) * 0.1:
                    report_lines.append("  - 奖励变化很小，可能已收敛，建议停止训练")
                
                if hasattr(self, 'intervention_history') and len(self.intervention_history) > 3:
                    report_lines.append("  - 干预次数较多，建议调整早熟检测参数")
        
        # 保存报告
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"奖励曲线分析报告已保存至: {report_path}")

    def get_task_assignments(self):
        """多次推理取最优值的任务分配方法"""
        self.model.eval()
        
        # 获取多次推理的参数
        n_inference_runs = getattr(self.config, 'RL_N_INFERENCE_RUNS', 10)
        inference_temperature = getattr(self.config, 'RL_INFERENCE_TEMPERATURE', 0.1)
        
        best_assignments = None
        best_reward = float('-inf')
        
        print(f"开始多次推理优化 (推理次数: {n_inference_runs})")
        
        for run in range(n_inference_runs):
            # 重置环境
            state = self.env.reset()
            assignments = {u.id: [] for u in self.env.uavs}
            done, step = False, 0
            total_reward = 0
            
            while not done and step < len(self.env.targets) * len(self.env.uavs):
                with torch.no_grad():
                    valid_mask = self._get_valid_action_mask()
                    if not valid_mask.any():
                        break
                    
                    # 添加温度参数以增加探索性
                    qs = self.model(torch.FloatTensor(state).unsqueeze(0).to(self.device)).squeeze(0)
                    qs[~valid_mask] = -float('inf')
                    
                    # 使用温度参数进行softmax采样
                    if inference_temperature > 0:
                        probs = torch.softmax(qs / inference_temperature, dim=0)
                        action_idx = torch.multinomial(probs, 1).item()
                    else:
                        action_idx = qs.argmax().item()
                
                action = self._index_to_action(action_idx)
                assignments[action[1]].append((action[0], action[2]))
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                step += 1
            
            # 评估当前推理结果
            if total_reward > best_reward:
                best_reward = total_reward
                best_assignments = assignments.copy()
                print(f"推理轮次 {run+1}: 发现更好的分配方案 (奖励: {best_reward:.2f})")
        
        self.model.train()
        print(f"多次推理完成，最优奖励: {best_reward:.2f}")
        return best_assignments
    

    def load_model(self, path):
        """(已修订) 从文件加载模型权重和训练历史，并处理PyTorch的FutureWarning。"""
        if os.path.exists(path):
            try:
                # 加载模型权重
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.target_model.load_state_dict(self.model.state_dict())
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                model_loaded = True
            except Exception:
                try:
                    state_dict = torch.load(path, map_location=self.device, weights_only=True)
                    self.model.load_state_dict(state_dict)
                    self.target_model.load_state_dict(self.model.state_dict())
                    model_loaded = True
                except Exception as e:
                    print(f"回退加载模型权重时出错: {e}")
                    return False

            if model_loaded:
                # 尝试加载训练历史
                import re
                match = re.search(r'ep_(\d+)_phrrt_(True|False)', path)
                if match:
                    episodes = match.group(1)
                    phrrt = match.group(2)
                    history_dir = os.path.dirname(path)
                    history_path = os.path.join(history_dir, f'training_history_ep_{episodes}_phrrt_{phrrt}.pkl')
                    if os.path.exists(history_path):
                        with open(history_path, 'rb') as f:
                            self.train_history = pickle.load(f)
                        print(f"调试: 已加载训练历史，奖励数据点数量: {len(self.train_history.get('episode_rewards', []))}")
                    else:
                        print(f"警告: 训练历史文件不存在: {history_path}")
                else:
                    print("警告: 无法从模型路径提取参数以加载训练历史")
                return True
        return False


# =============================================================================
# section 5: 核心业务流程
# =============================================================================
def _plan_single_leg(args):
    uav_id, start_pos, target_pos, start_heading, end_heading, obstacles, config = args
    planner = PHCurveRRTPlanner(start_pos, target_pos, start_heading, end_heading, obstacles, config); return uav_id, planner.plan()
# 此函数非常关键，直接觉得算法能够解除死锁、保证经济同步速度，而且速度还很快，协同分配资源的效率也很高。
def calculate_economic_sync_speeds(task_assignments, uavs, targets, graph, obstacles, config) -> Tuple[defaultdict, dict]:
    """(已更新) 计算经济同步速度，并返回未完成的任务以进行死锁检测。"""
    # 转换任务数据结构并补充资源消耗
    final_plan = defaultdict(list)
    uav_status = {u.id: {'pos': u.position, 'free_at': 0.0, 'heading': u.heading} for u in uavs}
    remaining_tasks = {uav_id: list(tasks) for uav_id, tasks in task_assignments.items()}; task_step_counter = defaultdict(lambda: 1)
    while any(v for v in remaining_tasks.values()):
        next_target_groups = defaultdict(list)
        for uav_id, tasks in remaining_tasks.items():
            if tasks: next_target_groups[tasks[0][0]].append({'uav_id': uav_id, 'phi_idx': tasks[0][1]})
        if not next_target_groups: break
        group_arrival_times = []
        for target_id, uav_infos in next_target_groups.items():
            target = next((t for t in targets if t.id == target_id), None)
            if not target: continue
            path_planners = {}
            for uav_info in uav_infos:
                uav_id = uav_info['uav_id']; args = (uav_id, uav_status[uav_id]['pos'], target.position, uav_status[uav_id]['heading'], graph.phi_set[uav_info['phi_idx']], obstacles, config)
                _, plan_result = _plan_single_leg(args)
                if plan_result: path_planners[uav_id] = {'path_points': plan_result[0], 'distance': plan_result[1]}
            time_windows = []
            for uav_info in uav_infos:
                uav_id = uav_info['uav_id']
                if uav_id not in path_planners: continue
                uav = next((u for u in uavs if u.id == uav_id), None)
                if not uav: continue
                distance = path_planners[uav_id]['distance']; free_at = uav_status[uav_id]['free_at']; t_min = free_at + (distance / uav.velocity_range[1]); t_max = free_at + (distance / uav.velocity_range[0]) if uav.velocity_range[0] > 0 else float('inf')
                t_econ = free_at + (distance / uav.economic_speed)
                time_windows.append({'uav_id': uav_id, 'phi_idx': uav_info['phi_idx'], 't_min': t_min, 't_max': t_max, 't_econ': t_econ})
            if not time_windows: continue
            sync_start = max(tw['t_min'] for tw in time_windows); sync_end = min(tw['t_max'] for tw in time_windows); is_feasible = sync_start <= sync_end + 1e-6
            final_sync_time = np.clip(np.median([tw['t_econ'] for tw in time_windows]), sync_start, sync_end) if is_feasible else sync_start
            group_arrival_times.append({'target_id': target_id, 'arrival_time': final_sync_time, 'uav_infos': time_windows, 'is_feasible': is_feasible, 'path_planners': path_planners})
        if not group_arrival_times: break
        next_event = min(group_arrival_times, key=lambda x: x['arrival_time']); target_pos = next(t.position for t in targets if t.id == next_event['target_id'])
        for uav_info in next_event['uav_infos']:
            uav_id = uav_info['uav_id']
            if uav_id not in next_event['path_planners']: continue
            uav, plan_data = next(u for u in uavs if u.id == uav_id), next_event['path_planners'][uav_id]; travel_time = next_event['arrival_time'] - uav_status[uav_id]['free_at']
            speed = (plan_data['distance'] / travel_time) if travel_time > 1e-6 else uav.velocity_range[1]
            final_plan[uav_id].append({'target_id': next_event['target_id'], 'start_pos': uav_status[uav_id]['pos'], 'speed': np.clip(speed, uav.velocity_range[0], uav.velocity_range[1]), 'arrival_time': next_event['arrival_time'], 'step': task_step_counter[uav_id], 'is_sync_feasible': next_event['is_feasible'], 'phi_idx': uav_info['phi_idx'], 'path_points': plan_data['path_points'], 'distance': plan_data['distance']})
            task_step_counter[uav_id] += 1; uav_status[uav_id].update(pos=target_pos, free_at=next_event['arrival_time'], heading=graph.phi_set[uav_info['phi_idx']])
            if remaining_tasks.get(uav_id): remaining_tasks[uav_id].pop(0)
    # 应用协同贪婪资源分配策略以匹配可视化逻辑
    temp_uav_resources = {u.id: u.initial_resources.copy().astype(float) for u in uavs}
    temp_target_resources = {t.id: t.resources.copy().astype(float) for t in targets}

    # 按事件分组任务
    events = defaultdict(list)
    for uav_id, tasks in final_plan.items():
        for task in tasks:
            event_key = (task['arrival_time'], task['target_id'])
            # 将无人机ID和任务引用存入对应的事件组
            events[event_key].append({'uav_id': uav_id, 'task_ref': task})
    
    # 按时间顺序处理事件
    for event_key in sorted(events.keys()):
        arrival_time, target_id = event_key
        collaborating_tasks = events[event_key]
        target_remaining = temp_target_resources[target_id].copy()
        
        # 分配资源
        for item in collaborating_tasks:
            uav_id = item['uav_id']
            task = item['task_ref']
            uav_resources = temp_uav_resources[uav_id]
            
            if np.all(target_remaining < 1e-6):
                task['resource_cost'] = np.zeros_like(uav_resources)
                continue
                
            contribution = np.minimum(target_remaining, uav_resources)
            task['resource_cost'] = contribution
            temp_uav_resources[uav_id] -= contribution
            target_remaining -= contribution
            
        temp_target_resources[target_id] = target_remaining

    return final_plan, remaining_tasks



# [新增] 从批处理测试器导入评估函数，用于对单个方案进行性能评估
from evaluate import evaluate_plan

def calibrate_resource_assignments(task_assignments, uavs, targets):
    """
    校准资源分配，移除无效的任务分配。
    
    Args:
        task_assignments: 原始任务分配
        uavs: 无人机列表
        targets: 目标列表
    
    Returns:
        校准后的任务分配
    """
    print("正在校准资源分配...")
    
    # 创建资源状态副本
    uav_resources = {u.id: u.initial_resources.copy().astype(float) for u in uavs}
    target_needs = {t.id: t.resources.copy().astype(float) for t in targets}
    
    # 按时间顺序处理任务分配（这里简化处理，按无人机ID顺序）
    calibrated_assignments = {u.id: [] for u in uavs}
    
    for uav_id in sorted(task_assignments.keys()):
        uav_tasks = task_assignments[uav_id]
        for target_id, phi_idx in uav_tasks:
            # 检查目标是否还需要资源
            if not np.any(target_needs[target_id] > 1e-6):
                print(f"警告: UAV {uav_id} 被分配到已满足的目标 {target_id}，跳过此分配")
                continue
            
            # 检查无人机是否还有资源
            if not np.any(uav_resources[uav_id] > 1e-6):
                print(f"警告: UAV {uav_id} 资源已耗尽，跳过后续分配")
                break
            
            # 计算实际贡献
            contribution = np.minimum(uav_resources[uav_id], target_needs[target_id])
            
            # 只有当有实际贡献时才保留此分配
            if np.any(contribution > 1e-6):
                calibrated_assignments[uav_id].append((target_id, phi_idx))
                uav_resources[uav_id] -= contribution
                target_needs[target_id] -= contribution
                print(f"UAV {uav_id} -> 目标 {target_id}: 贡献 {contribution}")
            else:
                print(f"警告: UAV {uav_id} 对目标 {target_id} 无有效贡献，跳过此分配")
    
    # 统计校准结果
    original_count = sum(len(tasks) for tasks in task_assignments.values())
    calibrated_count = sum(len(tasks) for tasks in calibrated_assignments.values())
    removed_count = original_count - calibrated_count
    
    print(f"资源分配校准完成:")
    print(f"  原始分配数量: {original_count}")
    print(f"  校准后数量: {calibrated_count}")
    print(f"  移除无效分配: {removed_count}")
    
    return calibrated_assignments


def visualize_task_assignments(final_plan, uavs, targets, obstacles, config, scenario_name, training_time, plan_generation_time,
                             save_plot=True, show_plot=False, save_report=False, deadlocked_tasks=None, evaluation_metrics=None):
    """(已更新并修复资源计算bug) 可视化任务分配方案。"""
    
    # [增加协同事件分析] 在报告中加入事件说明，解释资源竞争
    report_content = f'"""---------- {scenario_name} 执行报告 ----------\n\n'

    # [二次修复] 采用"协同贪婪"策略精确模拟资源消耗
    temp_uav_resources = {u.id: u.initial_resources.copy().astype(float) for u in uavs}
    temp_target_resources = {t.id: t.resources.copy().astype(float) for t in targets}

    # 1. 按"事件"（同一时间、同一目标）对所有步骤进行分组
    events = defaultdict(list)
    for uav_id, tasks in final_plan.items():
        for task in tasks:
            event_key = (task['arrival_time'], task['target_id'])
            # 将无人机ID和任务引用存入对应的事件组
            events[event_key].append({'uav_id': uav_id, 'task_ref': task})
    
    # 2. 按时间顺序（事件发生的顺序）对事件进行排序
    sorted_event_keys = sorted(events.keys())

    # [新增] 协同事件日志
    collaboration_log = "\n\n 协同事件日志 (揭示资源竞争):\n ------------------------------------\n"

    # 3. 按事件顺序遍历，处理每个协作事件
    for event_key in sorted_event_keys:
        arrival_time, target_id = event_key
        collaborating_steps = events[event_key]
        
        target_remaining_need_before = temp_target_resources[target_id].copy()
        collaboration_log += f" * 事件: 在 t={arrival_time:.2f}s, 无人机(UAVs) {', '.join([str(s['uav_id']) for s in collaborating_steps])} 到达 目标 {target_id}\n"
        collaboration_log += f"   - 目标初始需求: {target_remaining_need_before}\n"

        # 4. 在事件内部，让每个协作者依次、尽力地贡献资源
        for step in collaborating_steps:
            uav_id = step['uav_id']
            task = step['task_ref']

            uav_available_resources = temp_uav_resources[uav_id]
            actual_contribution = np.minimum(target_remaining_need_before, uav_available_resources)
            
            if np.all(actual_contribution < 1e-6):
                task['resource_cost'] = np.zeros_like(uav_available_resources)
                collaboration_log += f"     - UAV {uav_id} 尝试贡献，但目标需求已满足。贡献: [0. 0.]\n"
                continue

            temp_uav_resources[uav_id] -= actual_contribution
            target_remaining_need_before -= actual_contribution
            task['resource_cost'] = actual_contribution
            collaboration_log += f"     - UAV {uav_id} 贡献 {actual_contribution}, 剩余资源 {temp_uav_resources[uav_id]}\n"
            
        temp_target_resources[target_id] = target_remaining_need_before
        collaboration_log += f"   - 事件结束，目标剩余需求: {target_remaining_need_before}\n\n"

    """(已更新并修复资源计算bug) 可视化任务分配方案。"""
    
    # [二次修复] 采用"协同贪婪"策略精确模拟资源消耗
    temp_uav_resources = {u.id: u.initial_resources.copy().astype(float) for u in uavs}
    temp_target_resources = {t.id: t.resources.copy().astype(float) for t in targets}

    # 1. 按"事件"（同一时间、同一目标）对所有步骤进行分组
    events = defaultdict(list)
    for uav_id, tasks in final_plan.items():
        for task in tasks:
            event_key = (task['arrival_time'], task['target_id'])
            # 将无人机ID和任务引用存入对应的事件组
            events[event_key].append({'uav_id': uav_id, 'task_ref': task})
    
    # 2. 按时间顺序（事件发生的顺序）对事件进行排序
    sorted_event_keys = sorted(events.keys())

    # 3. 按事件顺序遍历，处理每个协作事件
    for event_key in sorted_event_keys:
        arrival_time, target_id = event_key
        collaborating_steps = events[event_key]
        
        target_remaining_need = temp_target_resources[target_id].copy()
        
        # 4. 在事件内部，让每个协作者依次、尽力地贡献资源
        for step in collaborating_steps:
            uav_id = step['uav_id']
            task = step['task_ref']

            if not np.any(target_remaining_need > 1e-6):
                task['resource_cost'] = np.zeros_like(temp_uav_resources[uav_id])
                continue

            uav_available_resources = temp_uav_resources[uav_id]
            actual_contribution = np.minimum(target_remaining_need, uav_available_resources)
            
            temp_uav_resources[uav_id] -= actual_contribution
            target_remaining_need -= actual_contribution
            
            task['resource_cost'] = actual_contribution
            
        temp_target_resources[target_id] = target_remaining_need


    # --- 后续的可视化和报告生成逻辑将使用上面计算出的精确 resource_cost ---
    fig, ax = plt.subplots(figsize=(22, 14)); ax.set_facecolor("#f0f0f0");
    for obs in obstacles: obs.draw(ax)

    target_collaborators_details = defaultdict(list)
    for uav_id, tasks in final_plan.items():
        for task in sorted(tasks, key=lambda x: x['step']):
            target_id = task['target_id']
            resource_cost = task.get('resource_cost', np.zeros_like(uavs[0].resources))
            target_collaborators_details[target_id].append({'uav_id': uav_id, 'arrival_time': task['arrival_time'], 'resource_cost': resource_cost})

    summary_text = ""
    if targets:
        satisfied_targets_count = 0; resource_types = len(targets[0].resources) if targets else 2
        total_demand_all = np.sum([t.resources for t in targets], axis=0)

        # --- [修订] 修复当 final_plan 为空时 np.sum 的计算错误 ---
        all_resource_costs = [d['resource_cost'] for details in target_collaborators_details.values() for d in details]
        if not all_resource_costs:
            total_contribution_all_for_summary = np.zeros(resource_types)
        else:
            total_contribution_all_for_summary = np.sum(all_resource_costs, axis=0)
        # --- 修订结束 ---

        for t in targets:
            current_target_contribution_sum = np.sum([d['resource_cost'] for d in target_collaborators_details.get(t.id, [])], axis=0)
            if np.all(current_target_contribution_sum >= t.resources - 1e-5): satisfied_targets_count += 1
        num_targets = len(targets); satisfaction_rate_percent = (satisfied_targets_count / num_targets * 100) if num_targets > 0 else 100
        total_demand_safe = total_demand_all.copy(); total_demand_safe[total_demand_safe == 0] = 1e-6
        overall_completion_rate_percent = np.mean(np.minimum(total_contribution_all_for_summary, total_demand_all) / total_demand_safe) * 100
        summary_text = (f"总体资源满足情况:\n--------------------------\n- 总需求: {np.array2string(total_demand_all, formatter={'float_kind':lambda x: '%.0f' % x})}\n- 总贡献: {np.array2string(total_contribution_all_for_summary, formatter={'float_kind':lambda x: '%.1f' % x})}\n- 已满足目标: {satisfied_targets_count} / {num_targets} ({satisfaction_rate_percent:.1f}%)\n- 资源完成率: {overall_completion_rate_percent:.1f}%")
    
    # ... (函数其余的可视化和报告生成代码未变) ...
    ax.scatter([u.position[0] for u in uavs], [u.position[1] for u in uavs], c='blue', marker='s', s=150, label='无人机起点', zorder=5, edgecolors='black')
    for u in uavs:
        ax.annotate(f"UAV{u.id}", (u.position[0], u.position[1]), fontsize=12, fontweight='bold', xytext=(0, -25), textcoords='offset points', ha='center', va='top')
        ax.annotate(f"初始: {np.array2string(u.initial_resources, formatter={'float_kind': lambda x: f'{x:.0f}'})}", (u.position[0], u.position[1]), fontsize=8, xytext=(15, 10), textcoords='offset points', ha='left', color='navy')
    ax.scatter([t.position[0] for t in targets], [t.position[1] for t in targets], c='red', marker='o', s=150, label='目标', zorder=5, edgecolors='black')
    for t in targets:
        demand_str = np.array2string(t.resources, formatter={'float_kind': lambda x: "%.0f" % x}); annotation_text = f"目标 {t.id}\n总需求: {demand_str}\n------------------"
        total_contribution = np.sum([d['resource_cost'] for d in target_collaborators_details.get(t.id, [])], axis=0)
        details_text = sorted(target_collaborators_details.get(t.id, []), key=lambda x: x['arrival_time'])
        if not details_text: annotation_text += "\n未分配无人机"
        else:
            for detail in details_text: annotation_text += f"\nUAV {detail['uav_id']} (T:{detail['arrival_time']:.1f}s) 贡献:{np.array2string(detail['resource_cost'], formatter={'float_kind': lambda x: '%.1f' % x})}"
        if np.all(total_contribution >= t.resources - 1e-5): satisfaction_str, bbox_color = "[OK] 需求满足", 'lightgreen'
        else: satisfaction_str, bbox_color = "[NG] 资源不足", 'mistyrose'
        annotation_text += f"\n------------------\n状态: {satisfaction_str}"
        ax.annotate(f"T{t.id}", (t.position[0], t.position[1]), fontsize=12, fontweight='bold', xytext=(0, 18), textcoords='offset points', ha='center', va='bottom')
        ax.annotate(annotation_text, (t.position[0], t.position[1]), fontsize=7, xytext=(15, -15), textcoords='offset points', ha='left', va='top', bbox=dict(boxstyle='round,pad=0.4', fc=bbox_color, ec='black', alpha=0.9, lw=0.5), zorder=8)

    colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(uavs))) if uavs else []; uav_color_map = {u.id: colors[i] for i, u in enumerate(uavs)}
    for uav_id, tasks in final_plan.items():
        uav_color = uav_color_map.get(uav_id, 'gray'); temp_resources = next(u for u in uavs if u.id == uav_id).initial_resources.copy().astype(float)
        for task in sorted(tasks, key=lambda x: x['step']):
            path_points = task.get('path_points')
            if path_points is not None and len(path_points) > 1:
                ax.plot(path_points[:, 0], path_points[:, 1], color=uav_color, linestyle='-' if task['is_sync_feasible'] else '--', linewidth=2, alpha=0.9, zorder=3)
                num_pos = path_points[int(len(path_points) * 0.45)]; ax.text(num_pos[0], num_pos[1], str(task['step']), color='white', backgroundcolor=uav_color, ha='center', va='center', fontsize=9, fontweight='bold', bbox=dict(boxstyle='circle,pad=0.2', fc=uav_color, ec='none'), zorder=4)
                resource_cost = task.get('resource_cost', np.zeros_like(temp_resources))
                temp_resources -= resource_cost
                resource_annotation_pos = path_points[int(len(path_points) * 0.85)]; remaining_res_str = f"R: {np.array2string(temp_resources.clip(0), formatter={'float_kind': lambda x: f'{x:.0f}'})}"
                ax.text(resource_annotation_pos[0], resource_annotation_pos[1], remaining_res_str, color=uav_color, backgroundcolor='white', ha='center', va='center', fontsize=7, fontweight='bold', bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=uav_color, alpha=0.8, lw=0.5), zorder=7)

    deadlock_summary_text = ""
    if deadlocked_tasks and any(deadlocked_tasks.values()):
        deadlock_summary_text += "!!! 死锁检测 !!!\n--------------------------\n以下无人机未能完成其任务序列，可能陷入死锁：\n"
        for uav_id, tasks in deadlocked_tasks.items():
            if tasks: deadlock_summary_text += f"- UAV {uav_id}: 等待执行 -> {' -> '.join([f'T{t[0]}' for t in tasks])}\n"
        deadlock_summary_text += ("-"*30) + "\n\n"
    report_header = f"---------- {scenario_name} 执行报告 ----------\n\n" + deadlock_summary_text
    if summary_text: report_header += summary_text + "\n" + ("-"*30) + "\n\n"
    
    # 添加评估指标到报告中
    if evaluation_metrics:
        report_header += "评估指标:\n--------------------------\n"
        for key, value in evaluation_metrics.items():
            # 特殊处理带归一化的指标
            if key in ['completion_rate', 'satisfied_targets_rate', 'sync_feasibility_rate', 'load_balance_score', 'resource_utilization_rate']:
                norm_value = evaluation_metrics.get(f'norm_{key}', 'N/A')
                if isinstance(norm_value, float):
                    print(f"  - {key}: {value:.4f} (归一化: {norm_value:.4f})")
                else:
                    print(f"  - {key}: {value:.4f} (归一化: {norm_value})")
        print("-" * 20)
    
    report_body_image = ""; report_body_file = ""
    for uav in uavs:
        uav_header = f"* 无人机 {uav.id} (初始资源: {np.array2string(uav.initial_resources, formatter={'float_kind': lambda x: f'{x:.0f}'})})\n"; report_body_image += uav_header; report_body_file += uav_header
        details = sorted(final_plan.get(uav.id, []), key=lambda x: x['step'])
        if not details: no_task_str = "  - 未分配任何任务\n"; report_body_image += no_task_str; report_body_file += no_task_str
        else:
            temp_resources_report = uav.initial_resources.copy().astype(float)
            for detail in details:
                resource_cost = detail.get('resource_cost', np.zeros_like(temp_resources_report))
                temp_resources_report -= resource_cost
                sync_status = "" if detail['is_sync_feasible'] else " (警告: 无法同步)"
                common_report_part = f"  {detail['step']}. 飞向目标 {detail['target_id']}{sync_status}:\n"; common_report_part += f"     - 飞行距离: {detail.get('distance', 0):.2f} m, 速度: {detail['speed']:.2f} m/s, 到达时间点: {detail['arrival_time']:.2f} s\n"
                common_report_part += f"     - 消耗资源: {np.array2string(resource_cost, formatter={'float_kind': lambda x: '%.1f' % x})}\n"; common_report_part += f"     - 剩余资源: {np.array2string(temp_resources_report.clip(0), formatter={'float_kind': lambda x: f'{x:.1f}'})}\n"
                report_body_image += common_report_part; report_body_file += common_report_part
                # 根据用户要求，报告中不输出路径点
                # path_points = detail.get('path_points')
                # if path_points is not None and len(path_points) > 0:
                #     points_per_line = 4; path_str_lines = []; line_buffer = []
                #     for p in path_points:
                #         line_buffer.append(f"({p[0]:.0f}, {p[1]:.0f})")
                #         if len(line_buffer) >= points_per_line: path_str_lines.append(" -> ".join(line_buffer)); line_buffer = []
                #     if line_buffer: path_str_lines.append(" -> ".join(line_buffer))
                #     report_body_file += "     - 路径坐标:\n"
                #     for line in path_str_lines: report_body_file += f"          {line}\n"
        report_body_image += "\n"; report_body_file += "\n"
    
    final_report_for_image = report_header + report_body_image; final_report_for_file = report_header + report_body_file
    plt.subplots_adjust(right=0.75); fig.text(0.77, 0.95, final_report_for_image, transform=plt.gcf().transFigure, ha="left", va="top", fontsize=9, bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', ec='grey', alpha=0.9))
    
    train_mode_str = '高精度' if config.USE_PHRRT_DURING_TRAINING else '快速近似'
    title_text = (
        f"多无人机任务分配与路径规划 - {scenario_name}\n"
        f"UAV: {len(uavs)}, 目标: {len(targets)}, 障碍: {len(obstacles)} | 模式: {train_mode_str}\n"
        f"模型训练耗时: {training_time:.2f}s | 方案生成耗时: {plan_generation_time:.2f}s"
    )
    ax.set_title(title_text, fontsize=12, fontweight='bold', pad=20)

    ax.set_xlabel("X坐标 (m)", fontsize=14); ax.set_ylabel("Y坐标 (m)", fontsize=14); ax.legend(loc="lower left"); ax.grid(True, linestyle='--', alpha=0.5, zorder=0); ax.set_aspect('equal', adjustable='box')
    
    if save_plot:
        output_dir = "output/images"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        clean_scenario_name = scenario_name.replace(' ', '_').replace(':', '')
        base_filename = f"{clean_scenario_name}_{timestamp}"
        img_filepath = os.path.join(output_dir, f"{base_filename}.png")
        plt.savefig(img_filepath, dpi=300)
        print(f"结果图已保存至: {img_filepath}")
        
        if save_report:
            report_dir = "output/reports"
            os.makedirs(report_dir, exist_ok=True)
            report_filepath = os.path.join(report_dir, f"{base_filename}.txt")
            try:
                with open(report_filepath, 'w', encoding='utf-8') as f:
                    f.write(final_report_for_file)
                print(f"任务分配报告已保存至: {report_filepath}")
            except Exception as e:
                print(f"错误：无法保存任务报告至 {report_filepath}. 原因: {e}")
                
    if show_plot:
        plt.show()
    plt.close(fig) # 确保无论是否显示，图形对象都被关闭以释放内存

# =============================================================================
# section 6: (新增) 辅助函数 & 主流程控制
# =============================================================================
import hashlib

def get_config_hash(config):
    """根据关键配置参数生成直观的配置标识字符串"""
    return (
        f"lr{config.LEARNING_RATE}_g{config.GAMMA}_"
        f"eps{config.EPSILON_END}-{config.EPSILON_DECAY}_"
        f"upd{config.TARGET_UPDATE_FREQ}_bs{config.BATCH_SIZE}_"
        f"phi{config.GRAPH_N_PHI}_phrrt{int(config.USE_PHRRT_DURING_TRAINING)}_"
        f"steps{config.EPISODES}"
    )

def _find_latest_checkpoint(model_path: str) -> Optional[str]:
    """查找最新的检查点文件"""
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        return None
    
    checkpoint_files = []
    for file in os.listdir(model_dir):
        if file.startswith('model_checkpoint_ep_') and file.endswith('.pth'):
            checkpoint_files.append(os.path.join(model_dir, file))
    
    if not checkpoint_files:
        return None
    
    # 按文件名中的轮次数排序
    checkpoint_files.sort(key=lambda x: int(x.split('_ep_')[1].split('.')[0]))
    return checkpoint_files[-1] if checkpoint_files else None

def _get_trained_episodes(model_path: str) -> int:
    """获取已训练的轮次数"""
    try:
        # 尝试从训练历史文件中获取
        history_dir = os.path.dirname(model_path)
        history_files = [f for f in os.listdir(history_dir) if f.startswith('training_history_')]
        
        if history_files:
            # 从最新的历史文件中提取轮次数
            latest_history = max(history_files, key=lambda x: os.path.getmtime(os.path.join(history_dir, x)))
            with open(os.path.join(history_dir, latest_history), 'rb') as f:
                import pickle
                history = pickle.load(f)
                return len(history.get('episode_rewards', []))
        
        # 如果无法从历史文件获取，尝试从检查点文件名获取
        checkpoint_path = _find_latest_checkpoint(model_path)
        if checkpoint_path:
            episode_str = checkpoint_path.split('_ep_')[1].split('.')[0]
            return int(episode_str)
        
        return 0
    except Exception as e:
        print(f"获取已训练轮次数时出错: {e}")
        return 0

def run_scenario(config, base_uavs, base_targets, obstacles, scenario_name, 
                 save_visualization=True, show_visualization=True, save_report=False,
                 force_retrain=False, incremental_training=False):
    """
    (已重构) 运行一个完整的场景测试，支持增量训练和自适应训练。
    
    Args:
        force_retrain: 强制重新训练，忽略已存在的模型
        incremental_training: 增量训练模式，在现有模型基础上继续训练
    """
    config_hash = get_config_hash(config)
    model_path = os.path.join('output', 'models', scenario_name.replace(' ', '_'), config_hash, 'model.pth')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    print(f"\n{'='*25}\n   Running: {scenario_name} (Config Hash: {config_hash})\n{'='*25}")

    uavs = [UAV(u.id, u.position, u.heading, u.resources, u.max_distance, u.velocity_range, u.economic_speed) for u in base_uavs]
    targets = [Target(t.id, t.position, t.resources, t.value) for t in base_targets]
    if not uavs or not targets: 
        print("场景中缺少无人机或目标，跳过执行。")
        return {}, 0.0, {}

    graph = DirectedGraph(uavs, targets, n_phi=config.GRAPH_N_PHI)
    i_dim = len(UAVTaskEnv(uavs, targets, graph, obstacles, config).reset())
    o_dim = len(targets) * len(uavs) * graph.n_phi
    
    # 检查是否使用自适应训练系统
    use_adaptive_training = hasattr(config, 'USE_ADAPTIVE_TRAINING') and config.USE_ADAPTIVE_TRAINING
    
    if use_adaptive_training:
        from temp_code.adaptive_training_system import AdaptiveGraphRLSolver
        solver = AdaptiveGraphRLSolver(uavs, targets, graph, obstacles, i_dim, 256, o_dim, config)
    else:
        solver = GraphRLSolver(uavs, targets, graph, obstacles, i_dim, 256, o_dim, config)
    
    training_time = 0.0
    
    # 智能模型加载逻辑
    model_loaded = False
    checkpoint_path = None
    
    if not force_retrain:
        # 尝试加载完整模型
        model_loaded = solver.load_model(model_path)
        
        if not model_loaded and incremental_training:
            # 尝试找到最新的检查点
            checkpoint_path = _find_latest_checkpoint(model_path)
            if checkpoint_path:
                print(f"找到检查点: {checkpoint_path}")
                if hasattr(solver, '_load_checkpoint'):
                    solver._load_checkpoint(checkpoint_path)
                    model_loaded = True
                    print(f"从检查点恢复训练")
    
    # 训练决策逻辑
    if config.RUN_TRAINING:
        if model_loaded and not force_retrain:
            if incremental_training:
                print(f"找到已训练模型，将进行增量训练... ({model_path})")
                # 增量训练：在现有基础上增加训练轮次
                additional_episodes = config.EPISODES - _get_trained_episodes(model_path)
                if additional_episodes > 0:
                    print(f"将在现有基础上增加 {additional_episodes} 轮训练")
                    training_time = solver.train(episodes=additional_episodes, patience=config.PATIENCE, 
                                               log_interval=config.LOG_INTERVAL, model_save_path=model_path)
                else:
                    print("模型已训练完成，无需额外训练")
            else:
                print(f"找到已训练模型，将重新训练... ({model_path})")
                training_time = solver.train(episodes=config.EPISODES, patience=config.PATIENCE, 
                                           log_interval=config.LOG_INTERVAL, model_save_path=model_path)
        else:
            print(f"开始新的训练... ({model_path})")
            training_time = solver.train(episodes=config.EPISODES, patience=config.PATIENCE, 
                                       log_interval=config.LOG_INTERVAL, model_save_path=model_path)
    elif not model_loaded:
        print(f"警告: 跳过训练，但未找到预训练模型 ({model_path})。任务分配将基于未训练的模型。")
    else:
        print(f"已加载预训练模型 ({model_path})，跳过训练阶段。")

    print(f"----- [阶段 2: 生成最终方案 ({scenario_name} @ {config_hash})] -----")
    # [已修订] 移除多余的二次加载逻辑。此时solver中的模型已是最新状态（刚训练的或已加载的）。
    task_assignments = solver.get_task_assignments()
    print("获取的任务分配方案:", {k:v for k,v in task_assignments.items() if v})
    
    # [新增] 校准资源分配，移除无效分配
    task_assignments = calibrate_resource_assignments(task_assignments, uavs, targets)
    print("校准后的任务分配方案:", {k:v for k,v in task_assignments.items() if v})
    
    plan_generation_start_time = time.time()
    final_plan, deadlocked_tasks = calculate_economic_sync_speeds(task_assignments, uavs, targets, graph, obstacles, config)
    plan_generation_time = time.time() - plan_generation_start_time
    
    # [新增] 调用从batch_tester导入的评估函数对最终方案进行量化评估
    evaluation_metrics = None
    if final_plan:
        print(f"----- [阶段 3: 方案质量评估 ({scenario_name})] -----")
        # 直接调用评估函数，并传入所需参数
        evaluation_metrics = evaluate_plan(final_plan, uavs, targets, deadlocked_tasks)
        
        # 格式化并打印评估结果
        print("评估指标:")
        for key, value in evaluation_metrics.items():
            if key == 'satisfied_targets_rate':
                print(f"  - {key}: {value:.2f} (实际满足: {evaluation_metrics['satisfied_targets_count']}/{evaluation_metrics['total_targets']})")
            elif isinstance(value, float):
                print(f"  - {key}: {value:.2f}")
            else:
                print(f"  - {key}: {value}")
        print("-" * 20)
    
    # 添加训练收敛情况和早熟检测结果到评估指标中
    if evaluation_metrics is None:
        evaluation_metrics = {}
    
    if config.RUN_TRAINING:
        evaluation_metrics['early_maturity_detected'] = solver.early_stop_detected
        if solver.early_stop_detected:
            print("警告: 检测到训练过程中可能存在早熟问题，增加训练轮次可能无法显著提升效果")
    
    if save_visualization or show_visualization:
        visualize_task_assignments(final_plan, uavs, targets, obstacles, config, scenario_name, 
                                 training_time=training_time,
                                 plan_generation_time=plan_generation_time,
                                 save_plot=save_visualization, 
                                 save_report=save_report,
                                 deadlocked_tasks=deadlocked_tasks,
                                 evaluation_metrics=evaluation_metrics)

    # 评估代码已移至上方

    return final_plan, training_time, deadlocked_tasks


def main():
    """主函数，用于单独运行和调试一个默认场景。"""
    set_chinese_font(manual_font_path='C:/Windows/Fonts/simhei.ttf')
    config = Config()
    
    # 测试自适应训练系统
    config.USE_ADAPTIVE_TRAINING = True  # 启用自适应训练
    config.RUN_TRAINING = True  # 启用训练
    
    # 从scenarios模块加载战略价值陷阱场景
    from scenarios import get_strategic_trap_scenario
    strategic_uavs, strategic_targets, strategic_obstacles = get_strategic_trap_scenario(config.OBSTACLE_TOLERANCE)
   
    print("\n--- 测试战略价值陷阱场景 ---")
    config.EPISODES = 300
    run_scenario(config, strategic_uavs, strategic_targets, strategic_obstacles, 
                "战略价值陷阱场景测试", show_visualization=False, save_report=True,
                force_retrain=False, incremental_training=True)

    # print("\n--- 测试1: 1000 ---")
    # config.EPISODES = 1000
    # run_scenario(config, complex_uavs, complex_targets, complex_obstacles, 
    #             "自适应训练测试-基础", show_visualization=False, save_report=True,
    #             force_retrain=False,incremental_training=True)

if __name__ == "__main__":
    main()




