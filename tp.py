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

# 允许多个OpenMP库共存，解决某些环境下的冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# =============================================================================
# section 0: 全局配置类
# =============================================================================
class Config:
    """
    集中管理所有可配置的算法超参数，方便进行调整和实验。
    """
    def __init__(self):
        # --- 核心运行模式控制 ---
        self.RUN_TRAINING = True                        # 是否执行RL训练。若为False且有模型缓存，则直接加载。
        self.USE_PHRRT_DURING_TRAINING = False          # 训练时是否使用高精度PH-RRT计算距离。False会使用快速近似距离，极大提升训练速度。
        self.CLEAR_MODEL_CACHE_BEFORE_TRAINING = True   # 每次运行是否清空旧的模型缓存文件。
        # [修改] 将模型、图片、报告等输出文件统一存放在 output 文件夹下
        self.SAVED_MODEL_PATH = 'output/models/saved_model_final_efficient.pth' # 模型保存/加载路径。

        # --- 强化学习 (RL) 训练超参数 ---
        self.EPISODES = 250             # 训练的总轮次。
        self.LEARNING_RATE = 0.0005     # 优化器的学习率。
        self.GAMMA = 0.98               # 折扣因子，决定未来奖励的重要性。
        self.BATCH_SIZE = 128           # 每次从记忆库中采样的数量。
        self.MEMORY_SIZE = 20000        # 记忆库的最大容量。
        self.EPSILON_DECAY = 0.9995      # Epsilon的衰减率，用于探索-利用平衡。
        self.EPSILON_MIN = 0.1          # Epsilon的最小值。
        self.TARGET_UPDATE_FREQ = 10    # 目标网络更新的频率（每N轮更新一次）。
        self.LOAD_BALANCE_PENALTY = 0.3 # 负载均衡惩罚系数。
        self.PATIENCE = 30              # 早停耐心值，连续N轮奖励无提升则停止训练。
        self.LOG_INTERVAL = 10          # 每N轮打印一次训练日志。

        # --- 路径规划 (Path Planning) 参数 ---
        self.RRT_ITERATIONS = 1500      # RRT算法的最大迭代次数。
        self.RRT_STEP_SIZE = 75.0       # RRT树扩展的步长。
        self.MAX_REFINEMENT_ATTEMPTS = 15 # PH曲线路径平滑的最大尝试次数。
        self.OBSTACLE_TOLERANCE = 50.0  # 障碍物的安全容忍距离。

        # --- 图构建参数 ---
        self.GRAPH_N_PHI = 6            # 构建图时，每个目标节点的离散化接近角度数量。

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
        self.load_balance_penalty = config.LOAD_BALANCE_PENALTY; self.alliance_bonus = 10.0; self.use_phrrt_in_training = config.USE_PHRRT_DURING_TRAINING; self.reset()
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
        target_id, uav_id, phi_idx = action; target = next((t for t in self.targets if t.id == target_id), None); uav = next((u for u in self.uavs if u.id == uav_id), None)
        if not target or not uav: return self._get_state(), -100, True, {}
        actual_contribution = np.minimum(target.remaining_resources, uav.resources)
        if np.all(actual_contribution <= 0): return self._get_state(), -20, False, {}
        if self.use_phrrt_in_training:
            start_heading = uav.heading if not uav.task_sequence else self.graph.phi_set[uav.task_sequence[-1][1]]
            planner = PHCurveRRTPlanner(uav.current_position, target.position, start_heading, self.graph.phi_set[phi_idx], self.obstacles, self.config); plan_result = planner.plan()
            path_len = plan_result[1] if plan_result else np.linalg.norm(target.position - uav.current_position) * 10
        else:
            start_v = (-uav.id, None) if not uav.task_sequence else (uav.task_sequence[-1][0], self.graph.phi_set[uav.task_sequence[-1][1]])
            end_v = (target.id, self.graph.phi_set[phi_idx]); path_len = self.graph.adjacency_matrix[self.graph.vertex_to_idx[start_v], self.graph.vertex_to_idx[end_v]]
        
        # [新增] 为贡献的资源本身提供正向奖励
        # 将贡献的两种资源相加，并乘以一个缩放因子作为奖励
        resource_contribution_reward = np.sum(actual_contribution) * 0.1 

        collaborators = {a[0] for a in target.allocated_uavs}; is_joining_alliance = len(collaborators) > 0 and uav_id not in collaborators
        uav.resources -= actual_contribution; target.remaining_resources -= actual_contribution
        if uav_id not in collaborators: target.allocated_uavs.append((uav_id, phi_idx))
        travel_time = path_len / uav.velocity_range[1]; reward = -travel_time; uav.task_sequence.append((target_id, phi_idx)); uav.current_distance += path_len; uav.current_position = target.position
        
        # [修改] 将资源贡献奖励加入回报计算
        reward += resource_contribution_reward

        if np.all(target.remaining_resources <= 0): reward += 50
        if is_joining_alliance: reward += self.alliance_bonus
        imbalance_penalty = np.var([len(u.task_sequence) for u in self.uavs]) * self.load_balance_penalty if self.uavs else 0
        done = all(np.all(t.remaining_resources <= 0) for t in self.targets); completion_bonus = 500 if done else 0
        final_reward = reward - imbalance_penalty + completion_bonus
        return self._get_state(), final_reward, done, {}


# =============================================================================
# section 4: 强化学习求解器
# =============================================================================
class GNN(nn.Module):
    """一个简单的全连接神经网络，用作Q值函数逼近器"""
    def __init__(self, i_dim, h_dim, o_dim): super(GNN, self).__init__(); self.l = nn.Sequential(nn.Linear(i_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, o_dim))
    def forward(self, x): return self.l(x)

class GraphRLSolver:
    """使用深度Q网络（DQN）的强化学习求解器"""
    def __init__(self, uavs, targets, graph, obstacles, i_dim, h_dim, o_dim, config):
        self.config, self.graph = config, graph; self.env = UAVTaskEnv(uavs, targets, graph, obstacles, config); self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GNN(i_dim, h_dim, o_dim).to(self.device); self.target_model = GNN(i_dim, h_dim, o_dim).to(self.device); self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE); self.memory = deque(maxlen=config.MEMORY_SIZE); self.epsilon, self.step_count = 1.0, 0; self.train_history = defaultdict(list)
        self.target_id_map = {t.id: i for i, t in enumerate(self.env.targets)}; self.uav_id_map = {u.id: i for i, u in enumerate(self.env.uavs)}
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
        for ep in tqdm(range(episodes), desc="Training"):
            state, done, total_reward = self.env.reset(), False, 0
            for _ in range(len(self.env.uavs) * len(self.env.targets)):
                valid_mask = self._get_valid_action_mask()
                if not valid_mask.any(): break
                if random.random() < self.epsilon: action_idx = random.choice(torch.where(valid_mask)[0]).item()
                else:
                    with torch.no_grad(): qs = self.model(torch.FloatTensor(state).unsqueeze(0).to(self.device)).squeeze(0); qs[~valid_mask] = -float('inf'); action_idx = qs.argmax().item()
                action = self._index_to_action(action_idx); next_state, reward, done, _ = self.env.step(action)
                self.remember(state, action, reward, next_state, done); state, total_reward = next_state, total_reward + reward; loss = self.replay(); self.step_count += 1
                if done: break
            self.train_history['episode_rewards'].append(total_reward)
            if (ep + 1) % log_interval == 0: tqdm.write(f"轮次 {ep+1}/{episodes} | 奖励: {total_reward:.2f} | Epsilon: {self.epsilon:.3f} | Loss: {loss or 0:.4f}")
            self.epsilon = max(self.config.EPSILON_MIN, self.epsilon * self.config.EPSILON_DECAY)
            if total_reward > best_reward:
                best_reward, early_stop_counter = total_reward, 0; torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, model_save_path)
            else: early_stop_counter += 1
            if early_stop_counter >= patience: print(f"Early stopping at epoch {ep+1}"); break
        return time.time() - start_time
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
        """(已修订) 从文件加载模型权重，并处理PyTorch的FutureWarning。"""
        if os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location=self.device, weights_only=False); self.model.load_state_dict(checkpoint['model_state_dict']); self.target_model.load_state_dict(self.model.state_dict()); self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']); return True
            except Exception:
                try:
                    state_dict = torch.load(path, map_location=self.device, weights_only=True); self.model.load_state_dict(state_dict); self.target_model.load_state_dict(self.model.state_dict()); return True
                except Exception as e: print(f"回退加载模型权重时出错: {e}"); return False
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
    final_plan = defaultdict(list); uav_status = {u.id: {'pos': u.position, 'free_at': 0.0, 'heading': u.heading} for u in uavs}
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
    return final_plan, remaining_tasks



def visualize_task_assignments(final_plan, uavs, targets, obstacles, config, scenario_name, training_time, plan_generation_time,
                             save_plot=True, show_plot=True, save_report=False, deadlocked_tasks=None):
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
                path_points = detail.get('path_points')
                if path_points is not None and len(path_points) > 0:
                    points_per_line = 4; path_str_lines = []; line_buffer = []
                    for p in path_points:
                        line_buffer.append(f"({p[0]:.0f}, {p[1]:.0f})")
                        if len(line_buffer) >= points_per_line: path_str_lines.append(" -> ".join(line_buffer)); line_buffer = []
                    if line_buffer: path_str_lines.append(" -> ".join(line_buffer))
                    report_body_file += "     - 路径坐标:\n"
                    for line in path_str_lines: report_body_file += f"          {line}\n"
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
                
    if show_plot: plt.show()
    plt.close(fig)

# =============================================================================
# section 6: 主流程控制
# =============================================================================
def run_scenario(config, base_uavs, base_targets, obstacles, scenario_name, 
                 save_visualization=True, show_visualization=True, save_report=False):
    """
    (已更新) 运行一个完整的场景测试，并对方案生成阶段进行计时。
    """
    print(f"\n{'='*25}\n   Running: {scenario_name}\n{'='*25}")

    # [修改] 自动创建模型保存目录，以防目录不存在导致torch.save失败
    model_dir = os.path.dirname(config.SAVED_MODEL_PATH)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        
    if config.CLEAR_MODEL_CACHE_BEFORE_TRAINING and os.path.exists(config.SAVED_MODEL_PATH):
        try: os.remove(config.SAVED_MODEL_PATH)
        except OSError as e: print(f"删除模型缓存失败: {e}")
    
    uavs = [UAV(u.id, u.position, u.heading, u.resources, u.max_distance, u.velocity_range, u.economic_speed) for u in base_uavs]
    targets = [Target(t.id, t.position, t.resources, t.value) for t in base_targets]
    if not uavs or not targets: print("场景中缺少无人机或目标，跳过执行。"); return {}, 0.0, {}

    graph = DirectedGraph(uavs, targets, n_phi=config.GRAPH_N_PHI)
    i_dim = len(UAVTaskEnv(uavs, targets, graph, obstacles, config).reset()); o_dim = len(targets) * len(uavs) * graph.n_phi
    solver = GraphRLSolver(uavs, targets, graph, obstacles, i_dim, 256, o_dim, config); training_time = 0.0; model_loaded = solver.load_model(config.SAVED_MODEL_PATH)
    if config.RUN_TRAINING or not model_loaded:
        if not model_loaded: print("未找到或无法加载预训练模型，将强制执行训练...")
        print(f"----- [阶段 1: 模型训练 ({scenario_name})] -----")
        training_time = solver.train(episodes=config.EPISODES, patience=config.PATIENCE, log_interval=config.LOG_INTERVAL, model_save_path=config.SAVED_MODEL_PATH)
    else: print("已加载预训练模型，跳过训练阶段。")
    print(f"----- [阶段 2: 生成最终方案 ({scenario_name})] -----")
    if not solver.load_model(config.SAVED_MODEL_PATH): print("警告: 无法加载模型以生成最终方案，结果可能为空或次优。")
    task_assignments = solver.get_task_assignments(); print("获取的任务分配方案:", {k:v for k,v in task_assignments.items() if v})
    
    plan_generation_start_time = time.time()
    final_plan, deadlocked_tasks = calculate_economic_sync_speeds(task_assignments, uavs, targets, graph, obstacles, config)
    plan_generation_time = time.time() - plan_generation_start_time
    
    if save_visualization or show_visualization:
        visualize_task_assignments(final_plan, uavs, targets, obstacles, config, scenario_name, 
                                 training_time=training_time,
                                 plan_generation_time=plan_generation_time,
                                 save_plot=save_visualization, 
                                 show_plot=show_visualization, 
                                 save_report=save_report,
                                 deadlocked_tasks=deadlocked_tasks)
                                 
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
    # run_scenario(config, small_uavs, small_targets, [], "预置场景（无障碍）", save_report=True)
    run_scenario(config, small_uavs, small_targets, small_obstacles, "预置场景（有障碍）", save_report=True)
    
    config.USE_PHRRT_DURING_TRAINING = True # 训练时使用PH-RRT算法
    run_scenario(config, small_uavs, small_targets, small_obstacles, "预置场景（有障碍）", save_report=True)

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




