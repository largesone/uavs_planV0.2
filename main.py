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
from typing import List, Dict, Tuple, Union
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

# --- 本地模块导入 ---
from entities import UAV, Target
from path_planning import PHCurveRRTPlanner
from scenarios import get_small_scenario, get_complex_scenario, get_new_experimental_scenario
from config import Config
from evaluate import evaluate_plan

# 允许多个OpenMP库共存，解决某些环境下的冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

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
            plt.rcParams['font.family'] = font_prop.get_name(); plt.rcParams['axes.unicode_minus'] = False
            print(f"成功加载手动指定的字体: {manual_font_path}"); return True
        except Exception as e: print(f"加载手动指定字体失败: {e}")
    if preferred_fonts is None: preferred_fonts = ['Source Han Sans SC', 'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'KaiTi', 'FangSong']
    try:
        for font in preferred_fonts:
            if findfont(FontProperties(family=font)):
                plt.rcParams["font.family"] = font; plt.rcParams['axes.unicode_minus'] = False
                print(f"已自动设置中文字体为: {font}"); return True
    except Exception: pass
    print("警告: 自动或手动设置中文字体失败。图片中的中文可能显示为方框。"); return False

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
        # 新增：用于检测早熟问题的变量
        self.best_reward_history = []
        self.loss_history = []
        self.early_stop_detected = False
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
    def train(self, episodes, patience, log_interval, model_save_path):
        start_time = time.time(); best_reward = -float('inf'); early_stop_counter = 0
        # 记录每个轮次的最佳奖励，用于检测早熟问题
        episode_rewards = []
        episode_losses = []
        
        for ep in tqdm(range(episodes), desc="Training"):
            state, done, total_reward = self.env.reset(), False, 0
            episode_loss = []
            
            for _ in range(len(self.env.uavs) * len(self.env.targets)):
                valid_mask = self._get_valid_action_mask()
                if not valid_mask.any(): break
                if random.random() < self.epsilon: action_idx = random.choice(torch.where(valid_mask)[0]).item()
                else:
                    with torch.no_grad(): qs = self.model(torch.FloatTensor(state).unsqueeze(0).to(self.device)).squeeze(0); qs[~valid_mask] = -float('inf'); action_idx = qs.argmax().item()
                action = self._index_to_action(action_idx); next_state, reward, done, _ = self.env.step(action)
                self.remember(state, action, reward, next_state, done); state, total_reward = next_state, total_reward + reward; 
                loss = self.replay()
                if loss is not None:
                    episode_loss.append(loss)
                self.step_count += 1
                if done: break
                
            # 记录本轮次的平均损失
            avg_loss = np.mean(episode_loss) if episode_loss else 0
            episode_losses.append(avg_loss)
            episode_rewards.append(total_reward)
            
            self.train_history['episode_rewards'].append(total_reward)
            if (ep + 1) % log_interval == 0: 
                tqdm.write(f"轮次 {ep+1}/{episodes} | 奖励: {total_reward:.2f} | Epsilon: {self.epsilon:.3f} | Loss: {avg_loss:.4f}")
                
                # 检测早熟问题 - 如果连续多轮奖励没有显著提升
                if ep >= 100:  # 至少训练100轮后才检测
                    recent_rewards = episode_rewards[-50:]  # 最近50轮的奖励
                    if max(recent_rewards) <= best_reward * 1.01:  # 最近50轮的最佳奖励没有比历史最佳提升1%
                        self.early_stop_detected = True
                        tqdm.write(f"警告: 检测到可能的早熟问题，最近50轮奖励无显著提升")
            
            self.epsilon = max(self.config.EPSILON_MIN, self.epsilon * self.config.EPSILON_DECAY)
            
            # 更新最佳奖励和早停计数器
            if total_reward > best_reward:
                best_reward, early_stop_counter = total_reward, 0
                self.best_reward_history.append((ep, total_reward))
                torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, model_save_path)
            else: 
                early_stop_counter += 1
            
            if early_stop_counter >= patience: 
                print(f"Early stopping at epoch {ep+1}")
                break
        
        # 保存训练历史数据，用于绘制收敛图
        self.train_history['episode_rewards'] = episode_rewards
        self.train_history['episode_losses'] = episode_losses
        self.train_history['best_rewards'] = self.best_reward_history
        
        # 新增: 保存训练历史到pickle文件
        history_dir = os.path.dirname(model_save_path)
        if not os.path.exists(history_dir):
            os.makedirs(history_dir)
        
        # 从模型保存路径中提取参数信息
        import re
        # 修改正则表达式以匹配实际的路径格式
        match = re.search(r'steps(\d+)', model_save_path)
        if match:
            episodes = match.group(1)
            phrrt = 'True' if config.USE_PHRRT_DURING_TRAINING else 'False'
            print(f"调试: 提取到参数 - episodes={episodes}, phrrt={phrrt}")
            history_path = os.path.join(history_dir, f'training_history_ep_{episodes}_phrrt_{phrrt}.pkl')
            # 确认train_history内容
            print(f"调试: 训练历史包含键: {self.train_history.keys()}, 奖励数据点数量: {len(self.train_history.get('episode_rewards', []))}")
            with open(history_path, 'wb') as f:
                pickle.dump(self.train_history, f)
            print(f"训练历史已保存至: {history_path}")
        else:
            print(f"警告: 未能从模型路径提取episodes参数，模型路径: {model_save_path}")
            # 使用默认参数保存
            history_path = os.path.join(history_dir, f'training_history_ep_{episodes}_phrrt_{config.USE_PHRRT_DURING_TRAINING}.pkl')
            with open(history_path, 'wb') as f:
                pickle.dump(self.train_history, f)
            print(f"训练历史已保存至: {history_path}")
        
        # 绘制并保存训练收敛图
        self._plot_convergence(model_save_path)
        
        return time.time() - start_time
        
    def _plot_convergence(self, model_save_path):
        """绘制训练收敛情况图表"""
        # 创建保存路径
        save_dir = os.path.dirname(model_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 创建图表
        plt.figure(figsize=(12, 10))
        
        # 1. 奖励收敛图
        plt.subplot(2, 1, 1)
        rewards = self.train_history['episode_rewards']
        plt.plot(rewards, label='每轮奖励')
        
        # 添加移动平均线
        window_size = min(50, len(rewards))
        if window_size > 0:
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(rewards)), moving_avg, 'r', label=f'{window_size}轮移动平均')
        
        # 标记最佳奖励点
        best_rewards = self.train_history['best_rewards']
        if best_rewards:
            best_x, best_y = zip(*best_rewards)
            plt.scatter(best_x, best_y, c='green', marker='*', s=100, label='最佳奖励')
        
        plt.title('训练奖励收敛情况')
        plt.xlabel('训练轮次')
        plt.ylabel('奖励值')
        plt.legend()
        plt.grid(True)
        
        # 2. 损失收敛图
        plt.subplot(2, 1, 2)
        losses = self.train_history['episode_losses']
        plt.plot(losses, label='每轮损失')
        
        # 添加移动平均线
        window_size = min(50, len(losses))
        if window_size > 0 and any(losses):
            moving_avg = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(losses)), moving_avg, 'r', label=f'{window_size}轮移动平均')
        
        plt.title('训练损失收敛情况')
        plt.xlabel('训练轮次')
        plt.ylabel('损失值')
        plt.legend()
        plt.grid(True)
        
        # 添加早熟检测结果
        if self.early_stop_detected:
            plt.figtext(0.5, 0.01, "警告: 检测到可能的早熟问题，训练轮次增加但效果无显著提升", 
                       ha='center', color='red', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))
        
        # 保存图表
        plt.tight_layout()
        convergence_plot_path = os.path.join(save_dir, 'training_convergence.png')
        plt.savefig(convergence_plot_path)
        plt.close()
        print(f"训练收敛情况图已保存至: {convergence_plot_path}")

    def get_task_assignments(self):
        self.model.eval(); state = self.env.reset(); assignments = {u.id: [] for u in self.env.uavs}; done, step = False, 0
        while not done and step < len(self.env.targets) * len(self.env.uavs):
            with torch.no_grad():
                valid_mask = self._get_valid_action_mask();
                if not valid_mask.any(): break
                qs = self.model(torch.FloatTensor(state).unsqueeze(0).to(self.device)).squeeze(0); qs[~valid_mask] = -float('inf'); action_idx = qs.argmax().item()
            action = self._index_to_action(action_idx); assignments[action[1]].append((action[0], action[2])); state, _, done, _ = self.env.step(action); step += 1
        self.model.train(); return assignments
    

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

def visualize_task_assignments(final_plan, uavs, targets, obstacles, config, scenario_name, training_time, plan_generation_time,
                             save_plot=True, show_plot=False, save_report=False, deadlocked_tasks=None, evaluation_metrics=None):
    """(已更新并修复资源计算bug) 可视化任务分配方案。"""
    
    # [增加协同事件分析] 在报告中加入事件说明，解释资源竞争
    report_content = f'"""---------- {scenario_name} 执行报告 ----------\n\n'

    # [二次修复] 采用“协同贪婪”策略精确模拟资源消耗
    temp_uav_resources = {u.id: u.initial_resources.copy().astype(float) for u in uavs}
    temp_target_resources = {t.id: t.resources.copy().astype(float) for t in targets}

    # 1. 按“事件”（同一时间、同一目标）对所有步骤进行分组
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
    
    # [二次修复] 采用“协同贪婪”策略精确模拟资源消耗
    temp_uav_resources = {u.id: u.initial_resources.copy().astype(float) for u in uavs}
    temp_target_resources = {t.id: t.resources.copy().astype(float) for t in targets}

    # 1. 按“事件”（同一时间、同一目标）对所有步骤进行分组
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

def run_scenario(config, base_uavs, base_targets, obstacles, scenario_name, 
                 save_visualization=True, show_visualization=True, save_report=False):
    """
    (已重构) 运行一个完整的场景测试，根据场景和配置参数动态加载/保存模型。
    此版本修复了在生成方案前重复加载模型的逻辑问题。
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
    solver = GraphRLSolver(uavs, targets, graph, obstacles, i_dim, 256, o_dim, config)
    training_time = 0.0
    
    # 检查模型是否存在
    model_loaded = solver.load_model(model_path)

    if config.RUN_TRAINING:
        if model_loaded:
            print(f"找到配置 {config_hash} 的已训练模型，将重新训练... ({model_path})")
        else:
            print(f"未找到配置 {config_hash} 的模型，开始新的训练... ({model_path})")
        print(f"----- [阶段 1: 模型训练 ({scenario_name} @ {config_hash})] -----")
        training_time = solver.train(episodes=config.EPISODES, patience=config.PATIENCE, log_interval=config.LOG_INTERVAL, model_save_path=model_path)
    elif not model_loaded:
        print(f"警告: 跳过训练，但未找到预训练模型 ({model_path})。任务分配将基于未训练的模型。")
    else:
        print(f"已加载预训练模型 ({model_path})，跳过训练阶段。")

    print(f"----- [阶段 2: 生成最终方案 ({scenario_name} @ {config_hash})] -----")
    # [已修订] 移除多余的二次加载逻辑。此时solver中的模型已是最新状态（刚训练的或已加载的）。
    task_assignments = solver.get_task_assignments()
    print("获取的任务分配方案:", {k:v for k,v in task_assignments.items() if v})
    
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
        
    
    # 从scenarios模块加载预置场景数据
    small_uavs, small_targets, small_obstacles = get_small_scenario(config.OBSTACLE_TOLERANCE)
    #small_uavs, small_targets, small_obstacles = get_new_experimental_scenario(config.OBSTACLE_TOLERANCE)   

    print("\n" + "="*80)
    print(">>> 正在执行默认的预置场景测试 <<<")
    print("="*80)
    config.USE_PHRRT_DURING_PLANNING = False # 近似距离算法    
    run_scenario(config, small_uavs, small_targets, small_obstacles, "预置场景（有障碍）", show_visualization=False,save_report=True)
    
    # config.USE_PHRRT_DURING_TRAINING = True # 训练时使用PH-RRT算法
    # run_scenario(config, small_uavs, small_targets, small_obstacles, "预置场景（有障碍）",show_visualization=False, save_report=True)


    # 从scenarios模块加载大规模复杂场景数据
    # complex_uavs, complex_targets, complex_obstacles = get_complex_scenario(config.OBSTACLE_TOLERANCE)
    
    # print("\n" + "="*80)
    # print(">>> 正在执行默认的大规模复杂场景测试 <<<")
    # print("="*80)
    # 取消下面一行的注释以运行大规模场景
    # run_scenario(config, complex_uavs, complex_targets, complex_obstacles, "大规模复杂场景（有障碍）", save_report=True)

    print("\n===== [调试运行结束] =====")

if __name__ == "__main__":
    main()




