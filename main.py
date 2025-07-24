# -*- coding: utf-8 -*-
# 文件名: main_simplified.py
# 描述: 多无人机协同任务分配与路径规划的简化版本。
#      移除了重复的网络结构定义，保留核心功能。

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
from collections import deque, defaultdict
import os
import time
import pickle
import random
import json
from typing import List, Dict, Tuple, Union, Optional
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import torch.nn.functional as F

# TensorBoard支持 - 可选依赖 
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("警告: TensorBoard未安装，将跳过TensorBoard功能")
    print("安装命令: pip install tensorboard")
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False

# --- 本地模块导入 ---
from entities import UAV, Target
from path_planning import PHCurveRRTPlanner
from scenarios import get_balanced_scenario, get_small_scenario, get_complex_scenario, get_new_experimental_scenario, get_complex_scenario_v4, get_strategic_trap_scenario
from config import Config
from evaluate import evaluate_plan
from networks import create_network, get_network_info
from environment import UAVTaskEnv, DirectedGraph

# 允许多个OpenMP库共存，解决某些环境下的冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 设置控制台输出编码
if sys.platform.startswith('win'):
    import codecs
    try:
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    except AttributeError:
        # 如果detach方法不可用，跳过编码设置
        pass

# 初始化配置类
config = Config()

# JSON序列化辅助函数
def convert_numpy_types(obj):
    """递归转换numpy类型为Python原生类型，用于JSON序列化"""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif hasattr(obj, 'item'):  # 处理任何其他numpy标量类型
        return obj.item()
    else:
        return obj

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
                # print(f"已自动设置中文字体为: {font}")  # 移除字体设置输出
                return True
    except Exception:
        pass
    
    # print("警告: 自动或手动设置中文字体失败。图片中的中文可能显示为方框。")  # 简化输出
    return False

# =============================================================================
# section 2: 核心业务逻辑 - 强化学习求解器
# =============================================================================

# =============================================================================
# section 3: 强化学习求解器
# =============================================================================

# =============================================================================
# section 3.5: 优先经验回放缓冲区 (Prioritized Experience Replay)
# =============================================================================
class SumTree:
    """
    Sum Tree数据结构，用于高效的优先级采样
    支持O(log n)的更新和采样操作
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # 存储优先级的二叉树
        self.data = np.zeros(capacity, dtype=object)  # 存储实际经验数据
        self.write = 0  # 写入指针
        self.n_entries = 0  # 当前存储的经验数量
    
    def _propagate(self, idx, change):
        """向上传播优先级变化"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx, s):
        """根据累积优先级检索叶子节点"""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self):
        """返回总优先级"""
        return self.tree[0]
    
    def add(self, p, data):
        """添加新经验"""
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx, p):
        """更新优先级"""
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)
    
    def get(self, s):
        """根据累积优先级获取经验"""
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class PrioritizedReplayBuffer:
    """
    优先经验回放缓冲区
    
    核心思想：
    - 根据TD误差分配优先级，误差越大优先级越高
    - 使用重要性采样权重修正非均匀采样的偏差
    - 通过α控制优先级的影响程度，β控制重要性采样的强度
    """
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        """
        Args:
            capacity: 缓冲区容量
            alpha: 优先级指数，0表示均匀采样，1表示完全按优先级采样
            beta_start: 重要性采样权重的初始值
            beta_frames: β从beta_start线性增长到1.0的帧数
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.epsilon = 1e-6  # 防止优先级为0
        self.max_priority = 1.0  # 最大优先级
    
    def beta(self):
        """计算当前的β值（重要性采样权重强度）"""
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)
    
    def push(self, state, action, reward, next_state, done):
        """添加新经验，使用最大优先级"""
        experience = (state, action, reward, next_state, done)
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, experience)
    
    def sample(self, batch_size):
        """
        采样一个批次的经验
        
        Returns:
            batch: 经验批次
            indices: 在树中的索引
            weights: 重要性采样权重
        """
        batch = []
        indices = []
        weights = []
        priorities = []
        
        # 计算采样区间
        segment = self.tree.total() / batch_size
        
        for i in range(batch_size):
            # 在每个区间内随机采样
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            # 获取经验
            (idx, p, data) = self.tree.get(s)
            
            if data is not None:
                batch.append(data)
                indices.append(idx)
                priorities.append(p)
        
        # 计算重要性采样权重
        if len(priorities) > 0:
            # 计算概率
            probs = np.array(priorities) / self.tree.total()
            # 计算权重
            weights = (len(batch) * probs) ** (-self.beta())
            # 归一化权重
            weights = weights / weights.max()
        else:
            weights = np.ones(len(batch))
        
        self.frame += 1
        
        return batch, indices, weights
    
    def update_priorities(self, indices, priorities):
        """更新经验的优先级"""
        for idx, priority in zip(indices, priorities):
            # 确保优先级为正数
            priority = abs(priority) + self.epsilon
            # 更新最大优先级
            self.max_priority = max(self.max_priority, priority)
            # 应用α指数
            priority = priority ** self.alpha
            # 更新树中的优先级
            self.tree.update(idx, priority)
    
    def __len__(self):
        """返回当前存储的经验数量"""
        return self.tree.n_entries

# =============================================================================
# section 4: 简化的强化学习求解器
# =============================================================================
class GraphRLSolver:
    """简化的基于图的强化学习求解器 - 增强版本，支持TensorBoard监控"""
    def __init__(self, uavs, targets, graph, obstacles, i_dim, h_dim, o_dim, config, network_type="SimpleNetwork", tensorboard_dir=None):
        self.uavs = uavs
        self.targets = targets
        self.graph = graph
        self.obstacles = obstacles
        self.config = config
        self.network_type = network_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # TensorBoard支持 - 安全初始化
        self.tensorboard_dir = tensorboard_dir
        self.writer = None
        if tensorboard_dir and TENSORBOARD_AVAILABLE:
            try:
                self.writer = SummaryWriter(tensorboard_dir)
                print(f"TensorBoard日志将保存至: {tensorboard_dir}")
            except Exception as e:
                print(f"TensorBoard初始化失败: {e}")
        elif tensorboard_dir and not TENSORBOARD_AVAILABLE:
            print("TensorBoard未安装，跳过日志记录")
        
        # 创建网络
        self.policy_net = create_network(network_type, i_dim, h_dim, o_dim).to(self.device)
        self.target_net = create_network(network_type, i_dim, h_dim, o_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)
        
        # 使用优先经验回放缓冲区替代普通deque
        self.use_per = config.training_config.use_prioritized_replay
        if self.use_per:
            self.memory = PrioritizedReplayBuffer(
                capacity=config.MEMORY_CAPACITY,
                alpha=config.training_config.per_alpha,
                beta_start=config.training_config.per_beta_start,
                beta_frames=config.training_config.per_beta_frames
            )
            print(f"  - 优先经验回放: 启用 (α={config.training_config.per_alpha}, β_start={config.training_config.per_beta_start})")
        else:
            self.memory = deque(maxlen=config.MEMORY_CAPACITY)
            print("  - 优先经验回放: 禁用")
        self.epsilon = config.training_config.epsilon_start
        
        # 环境
        self.env = UAVTaskEnv(uavs, targets, graph, obstacles, config)
        
        # 动作映射
        self.target_id_map = {t.id: i for i, t in enumerate(self.env.targets)}
        self.uav_id_map = {u.id: i for i, u in enumerate(self.env.uavs)}
        
        # 训练参数 - 进一步优化的超参数
        self.epsilon_decay = 0.995  # 适中的衰减率，平衡探索与利用
        self.epsilon_min = 0.15     # 提高最小探索率，确保持续探索
        
        # 高级DQN技术
        self.use_double_dqn = True  # 启用Double DQN
        self.use_dueling_dqn = True # 启用Dueling DQN架构
        self.use_grad_clip = True   # 启用梯度裁剪
        self.use_prioritized_replay = True  # 启用优先经验回放
        
        # 学习率调整
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.95)
        self.grad_clip_norm = 0.5   # 更严格的梯度裁剪阈值
        
        # 训练统计
        self.step_count = 0
        self.update_count = 0
        
        print(f"初始化完成: {network_type} 网络")
        print(f"  - Double DQN: {'启用' if self.use_double_dqn else '禁用'}")
        print(f"  - 梯度裁剪: {'启用' if self.use_grad_clip else '禁用'}")
        print(f"  - 探索率衰减: {self.epsilon_decay}")
    
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
            # 临时切换到eval模式避免BatchNorm问题
            self.policy_net.eval()
            q_values = self.policy_net(state)
            self.policy_net.train()
            return q_values.max(1)[1].view(1, 1)
    
    def optimize_model(self):
        """
        从经验回放池中采样并优化模型 - 支持优先经验回放(PER)
        
        核心改进:
        1. 支持PER的带权重采样
        2. 计算TD误差并更新优先级
        3. 使用重要性采样权重修正偏差
        """
        if len(self.memory) < self.config.BATCH_SIZE:
            return
        
        # 根据是否使用PER选择不同的采样策略
        if self.use_per:
            # PER采样：获取经验、索引和重要性采样权重
            transitions, indices, weights = self.memory.sample(self.config.BATCH_SIZE)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            # 标准随机采样
            transitions = random.sample(self.memory, self.config.BATCH_SIZE)
            indices = None
            weights = torch.ones(self.config.BATCH_SIZE).to(self.device)
        
        # 解包批次数据
        batch = list(zip(*transitions))
        state_batch = torch.cat(batch[0])
        action_batch = torch.cat(batch[1])
        reward_batch = torch.cat(batch[2])
        next_states_batch = torch.cat(batch[3])
        done_batch = torch.tensor(batch[4], device=self.device, dtype=torch.bool)
        
        # 计算当前Q值
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # 计算目标Q值
        next_q_values = torch.zeros(self.config.BATCH_SIZE, device=self.device)
        
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: 使用策略网络选择动作，目标网络评估动作
                next_actions = self.policy_net(next_states_batch).max(1)[1].unsqueeze(1)
                next_q_values[~done_batch] = self.target_net(next_states_batch[~done_batch]).gather(1, next_actions[~done_batch]).squeeze(1)
            else:
                # 标准DQN
                next_q_values[~done_batch] = self.target_net(next_states_batch[~done_batch]).max(1)[0]
        
        expected_q_values = reward_batch + (self.config.GAMMA * next_q_values)
        
        # 计算TD误差（用于更新优先级）
        td_errors = (current_q_values.squeeze() - expected_q_values).detach()
        
        # 计算加权损失
        if self.use_per:
            # 使用重要性采样权重修正损失
            elementwise_loss = F.smooth_l1_loss(current_q_values, expected_q_values.unsqueeze(1), reduction='none')
            loss = (elementwise_loss.squeeze() * weights).mean()
        else:
            # 标准损失
            loss = F.smooth_l1_loss(current_q_values, expected_q_values.unsqueeze(1))
        
        # 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip_norm)
        
        self.optimizer.step()
        self.update_count += 1
        
        # 更新PER优先级
        if self.use_per and indices is not None:
            # 使用TD误差的绝对值作为新的优先级
            priorities = np.abs(td_errors.cpu().numpy()) + 1e-6
            self.memory.update_priorities(indices, priorities)
        
        # TensorBoard记录
        if self.writer:
            # 记录基础指标
            self.writer.add_scalar('Training/Loss', loss.item(), self.update_count)
            self.writer.add_scalar('Training/Mean_Q_Value', current_q_values.mean().item(), self.update_count)
            self.writer.add_scalar('Training/Mean_TD_Error', td_errors.abs().mean().item(), self.update_count)
            
            # 记录PER相关指标
            if self.use_per:
                self.writer.add_scalar('Training/PER_Beta', self.memory.beta(), self.update_count)
                self.writer.add_scalar('Training/Mean_IS_Weight', weights.mean().item(), self.update_count)
                self.writer.add_scalar('Training/Max_Priority', self.memory.max_priority, self.update_count)
            
            # 记录梯度信息
            total_norm = 0
            for p in self.policy_net.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            self.writer.add_scalar('Training/Gradient_Norm', total_norm, self.update_count)
        
        # 软更新目标网络
        if self.update_count % self.config.TARGET_UPDATE_FREQ == 0:
            tau = 0.01  # 软更新系数
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)
        
        # 自适应学习率调度
        if self.update_count % 500 == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'] * 0.95, 1e-5)
        
        return loss.item()
        
        # 记录训练指标到TensorBoard
        if self.writer:
            self.writer.add_scalar('Training/Loss', loss.item(), self.update_count)
            self.writer.add_scalar('Training/Q_Value_Mean', current_q_values.mean().item(), self.update_count)
            self.writer.add_scalar('Training/Q_Value_Std', current_q_values.std().item(), self.update_count)
            self.writer.add_scalar('Training/Target_Q_Mean', expected_q_values.mean().item(), self.update_count)
            self.writer.add_scalar('Training/Reward_Mean', reward_batch.mean().item(), self.update_count)
        
        return loss.item()
    
    def train(self, episodes, patience, log_interval, model_save_path):
        """训练模型 - 增强版本，支持TensorBoard监控和详细调试信息"""
        start_time = time.time()
        best_reward = float('-inf')
        patience_counter = 0
        
        # 初始化训练历史记录
        self.episode_rewards = []
        self.episode_losses = []
        self.epsilon_values = []
        self.completion_rates = []
        self.episode_steps = []
        self.memory_usage = []
        
        # 简化训练开始信息
        param_count = sum(p.numel() for p in self.policy_net.parameters())
        print(f"初始化 {self.network_type} 网络 (参数: {param_count:,}, 设备: {self.device})")
        
        for i_episode in tqdm(range(episodes), desc=f"训练进度 [{self.network_type}]"):
            state = self.env.reset()
            episode_reward = 0
            episode_loss = 0
            loss_count = 0
            episode_step_count = 0
            
            for step in range(self.env.max_steps):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action = self.select_action(state_tensor)
                
                next_state, reward, done, truncated, _ = self.env.step(action.item())
                episode_reward += reward
                episode_step_count += 1
                self.step_count += 1
                
                # 添加经验到回放缓冲区
                if self.use_per:
                    self.memory.push(
                        state_tensor,
                        action,
                        torch.tensor([reward], device=self.device),
                        torch.FloatTensor(next_state).unsqueeze(0).to(self.device),
                        done
                    )
                else:
                    self.memory.append((
                        state_tensor,
                        action,
                        torch.tensor([reward], device=self.device),
                        torch.FloatTensor(next_state).unsqueeze(0).to(self.device),
                        done
                    ))
                
                # 优化模型（每步都尝试优化）
                if len(self.memory) >= self.config.BATCH_SIZE:
                    loss = self.optimize_model()
                    if loss is not None:
                        episode_loss += loss
                        loss_count += 1
                
                state = next_state
                
                if done or truncated:
                    break
            
            # 更新目标网络
            if i_episode % self.config.training_config.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                if self.writer:
                    self.writer.add_scalar('Training/Target_Network_Update', 1, i_episode)
            
            # 衰减探索率 - 使用调整后的参数
            self.epsilon = max(self.epsilon_min, 
                             self.epsilon * self.epsilon_decay)
            
            # 记录训练历史
            avg_episode_loss = episode_loss / max(loss_count, 1)
            self.episode_rewards.append(episode_reward)
            self.episode_losses.append(avg_episode_loss)
            self.epsilon_values.append(self.epsilon)
            self.episode_steps.append(episode_step_count)
            self.memory_usage.append(len(self.memory))
            
            # 计算完成率 - 优化版本，综合考虑目标满足数量、总体资源满足率、资源利用率
            if self.env.targets:
                # 1. 目标满足数量比例
                completed_targets = sum(1 for target in self.env.targets if np.all(target.remaining_resources <= 0))
                target_satisfaction_rate = completed_targets / len(self.env.targets)
                
                # 2. 总体资源满足率
                total_demand = sum(np.sum(target.resources) for target in self.env.targets)
                total_remaining = sum(np.sum(target.remaining_resources) for target in self.env.targets)
                resource_satisfaction_rate = (total_demand - total_remaining) / total_demand if total_demand > 0 else 0
                
                # 3. 资源利用率
                total_initial_supply = sum(np.sum(uav.initial_resources) for uav in self.env.uavs)
                total_current_supply = sum(np.sum(uav.resources) for uav in self.env.uavs)
                resource_utilization_rate = (total_initial_supply - total_current_supply) / total_initial_supply if total_initial_supply > 0 else 0
                
                # 综合完成率 = 目标满足率(50%) + 资源满足率(30%) + 资源利用率(20%)
                completion_rate = (target_satisfaction_rate * 0.5 + 
                                 resource_satisfaction_rate * 0.3 + 
                                 resource_utilization_rate * 0.2)
            else:
                completion_rate = 0
            
            self.completion_rates.append(completion_rate)
            
            # 增强的TensorBoard记录
            if self.writer:
                # 基础指标
                self.writer.add_scalar('Episode/Reward', episode_reward, i_episode)
                self.writer.add_scalar('Episode/Loss', avg_episode_loss, i_episode)
                self.writer.add_scalar('Episode/Epsilon', self.epsilon, i_episode)
                self.writer.add_scalar('Episode/Completion_Rate', completion_rate, i_episode)
                self.writer.add_scalar('Episode/Steps', episode_step_count, i_episode)
                self.writer.add_scalar('Episode/Memory_Usage', len(self.memory), i_episode)
                
                # 移动平均指标
                if len(self.episode_rewards) >= 20:
                    recent_reward_avg = np.mean(self.episode_rewards[-20:])
                    recent_completion_avg = np.mean(self.completion_rates[-20:])
                    self.writer.add_scalar('Episode/Reward_MA20', recent_reward_avg, i_episode)
                    self.writer.add_scalar('Episode/Completion_Rate_MA20', recent_completion_avg, i_episode)
                
                # 收敛性指标
                if len(self.episode_rewards) >= 50:
                    recent_std = np.std(self.episode_rewards[-50:])
                    overall_std = np.std(self.episode_rewards)
                    stability_ratio = recent_std / overall_std if overall_std > 0 else 0
                    self.writer.add_scalar('Convergence/Stability_Ratio', stability_ratio, i_episode)
                    self.writer.add_scalar('Convergence/Recent_Std', recent_std, i_episode)
                
                # 早停相关指标
                self.writer.add_scalar('Training/Patience_Counter', patience_counter, i_episode)
                self.writer.add_scalar('Training/Best_Reward', best_reward, i_episode)
                
                # 网络权重和梯度分析
                if i_episode % (log_interval * 2) == 0:
                    for name, param in self.policy_net.named_parameters():
                        if param.grad is not None:
                            self.writer.add_histogram(f'Weights/{name}', param, i_episode)
                            self.writer.add_histogram(f'Gradients/{name}', param.grad, i_episode)
                            self.writer.add_scalar(f'Gradients/{name}_norm', param.grad.norm(), i_episode)
                
                # 学习率记录（如果使用调度器）
                if hasattr(self, 'optimizer'):
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.writer.add_scalar('Training/Learning_Rate', current_lr, i_episode)
            
            # 简化的训练进度输出 - 仅在关键节点输出
            if i_episode % (log_interval * 5) == 0 and i_episode > 0:  # 减少输出频率
                avg_reward = np.mean(self.episode_rewards[-log_interval:])
                avg_completion = np.mean(self.completion_rates[-log_interval:])
                
                print(f"训练进度 - Episode {i_episode:4d}: 平均奖励 {avg_reward:8.2f}, 完成率 {avg_completion:6.3f}")
            
            # 改进的早停检查 - 基于资源满足率和训练进度
            current_completion_rate = completion_rate
            
            # 1. 传统奖励早停（保留但提高阈值）
            if episode_reward > best_reward:
                best_reward = episode_reward
                patience_counter = 0
                self.save_model(model_save_path)
                if self.writer:
                    self.writer.add_scalar('Training/Best_Reward', best_reward, i_episode)
            else:
                patience_counter += 1
            
            # 2. 新增：基于资源满足率的早停准则
            should_early_stop = False
            early_stop_reason = ""
            
            # 确保至少训练总轮次的20%
            min_training_episodes = max(int(episodes * 0.2), 100)
            
            if i_episode >= min_training_episodes:
                # 检查最近50轮的平均完成率
                if len(self.completion_rates) >= 50:
                    recent_completion_avg = np.mean(self.completion_rates[-50:])
                    
                    # 如果资源满足率持续高于95%，可以早停
                    if recent_completion_avg >= 0.95:
                        should_early_stop = True
                        early_stop_reason = f"资源满足率达标 (平均: {recent_completion_avg:.3f})"
                    
                    # 检查收敛性：最近50轮的标准差很小
                    recent_completion_std = np.std(self.completion_rates[-50:])
                    if recent_completion_std < 0.02 and recent_completion_avg >= 0.85:
                        should_early_stop = True
                        early_stop_reason = f"完成率收敛 (平均: {recent_completion_avg:.3f}, 标准差: {recent_completion_std:.4f})"
            
            # 传统早停（提高patience阈值，避免过早停止）
            if patience_counter >= patience * 2:  # 将patience阈值翻倍
                should_early_stop = True
                early_stop_reason = f"奖励无改进超过 {patience * 2} 轮"
            
            if should_early_stop:
                print(f"早停触发于第 {i_episode} 回合: {early_stop_reason}")
                print(f"最佳奖励: {best_reward:.2f}, 最终完成率: {current_completion_rate:.3f}")
                break
        
        training_time = time.time() - start_time
        
        # 关闭TensorBoard writer
        if self.writer:
            self.writer.close()
        
        print(f"\n训练完成 - 耗时: {training_time:.2f}秒")
        print(f"训练统计:")
        print(f"  总回合数: {len(self.episode_rewards)}")
        print(f"  最佳奖励: {best_reward:.2f}")
        print(f"  最终完成率: {self.completion_rates[-1]:.3f} (综合目标满足、资源满足、资源利用率)")
        print(f"  最终探索率: {self.epsilon:.4f} (随机动作概率)")
        
        # 生成详细的收敛性分析
        self.generate_enhanced_convergence_analysis(model_save_path, i_episode)
        
        return training_time
    
    def get_convergence_metrics(self):
        """获取收敛性指标"""
        if not self.episode_rewards:
            return {}
        
        rewards = np.array(self.episode_rewards)
        
        # 计算收敛性指标
        metrics = {
            'final_reward': float(rewards[-1]),
            'max_reward': float(np.max(rewards)),
            'mean_reward': float(np.mean(rewards)),
            'reward_std': float(np.std(rewards)),
            'reward_improvement': float(rewards[-1] - rewards[0]) if len(rewards) > 1 else 0.0
        }
        
        # 收敛稳定性分析
        if len(rewards) > 100:
            recent_rewards = rewards[-50:]
            early_rewards = rewards[:50]
            
            metrics.update({
                'recent_mean': float(np.mean(recent_rewards)),
                'recent_std': float(np.std(recent_rewards)),
                'early_mean': float(np.mean(early_rewards)),
                'stability_ratio': float(np.std(recent_rewards) / np.std(rewards)) if np.std(rewards) > 0 else 0.0,
                'improvement_ratio': float((np.mean(recent_rewards) - np.mean(early_rewards)) / abs(np.mean(early_rewards))) if np.mean(early_rewards) != 0 else 0.0
            })
            
            # 判断收敛状态
            if metrics['stability_ratio'] < 0.3:
                metrics['convergence_status'] = 'converged'
            elif metrics['stability_ratio'] < 0.6:
                metrics['convergence_status'] = 'partially_converged'
            else:
                metrics['convergence_status'] = 'unstable'
        
        # 添加完成率相关指标
        if hasattr(self, 'completion_rates') and self.completion_rates:
            completion_rates = np.array(self.completion_rates)
            metrics.update({
                'final_completion_rate': float(completion_rates[-1]),
                'max_completion_rate': float(np.max(completion_rates)),
                'mean_completion_rate': float(np.mean(completion_rates)),
                'completion_rate_std': float(np.std(completion_rates)),
                'completion_improvement': float(completion_rates[-1] - completion_rates[0]) if len(completion_rates) > 1 else 0.0
            })
            
            # 完成率收敛性分析
            if len(completion_rates) > 50:
                recent_completion = completion_rates[-50:]
                metrics.update({
                    'recent_completion_mean': float(np.mean(recent_completion)),
                    'recent_completion_std': float(np.std(recent_completion)),
                    'completion_stability': float(np.std(recent_completion) / np.std(completion_rates)) if np.std(completion_rates) > 0 else 0.0
                })
        
        return metrics
    
    def generate_enhanced_convergence_analysis(self, model_save_path, final_episode):
        """生成增强的收敛性分析图表和报告"""
        if not self.episode_rewards:
            return
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建多子图布局
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle(f'{self.network_type} 网络训练收敛性分析', fontsize=16, fontweight='bold')
        
        episodes = range(1, len(self.episode_rewards) + 1)
        
        # 1. 奖励曲线和移动平均
        ax1 = axes[0, 0]
        ax1.plot(episodes, self.episode_rewards, alpha=0.6, color='lightblue', label='原始奖励')
        if len(self.episode_rewards) > 20:
            moving_avg = np.convolve(self.episode_rewards, np.ones(20)/20, mode='valid')
            ax1.plot(range(20, len(self.episode_rewards) + 1), moving_avg, 
                    color='red', linewidth=2, label='移动平均(20)')
        ax1.set_title('奖励收敛曲线')
        ax1.set_xlabel('训练轮次')
        ax1.set_ylabel('奖励值')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 损失曲线
        ax2 = axes[0, 1]
        if self.episode_losses:
            ax2.plot(episodes, self.episode_losses, color='orange', alpha=0.7)
            if len(self.episode_losses) > 20:
                loss_moving_avg = np.convolve(self.episode_losses, np.ones(20)/20, mode='valid')
                ax2.plot(range(20, len(self.episode_losses) + 1), loss_moving_avg, 
                        color='red', linewidth=2, label='移动平均(20)')
        ax2.set_title('损失收敛曲线')
        ax2.set_xlabel('训练轮次')
        ax2.set_ylabel('损失值')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 完成率曲线
        ax3 = axes[1, 0]
        if self.completion_rates:
            ax3.plot(episodes, self.completion_rates, color='green', alpha=0.7, label='完成率')
            if len(self.completion_rates) > 20:
                completion_moving_avg = np.convolve(self.completion_rates, np.ones(20)/20, mode='valid')
                ax3.plot(range(20, len(self.completion_rates) + 1), completion_moving_avg, 
                        color='red', linewidth=2, label='移动平均(20)')
            ax3.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='早停阈值(0.95)')
            ax3.axhline(y=0.85, color='orange', linestyle='--', alpha=0.7, label='收敛阈值(0.85)')
        ax3.set_title('完成率收敛曲线')
        ax3.set_xlabel('训练轮次')
        ax3.set_ylabel('完成率')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 探索率衰减
        ax4 = axes[1, 1]
        if self.epsilon_values:
            ax4.plot(episodes, self.epsilon_values, color='purple', alpha=0.8)
        ax4.set_title('探索率衰减曲线')
        ax4.set_xlabel('训练轮次')
        ax4.set_ylabel('探索率 (ε)')
        ax4.grid(True, alpha=0.3)
        
        # 5. 收敛性指标分析
        ax5 = axes[2, 0]
        if len(self.episode_rewards) > 100:
            # 计算滑动窗口标准差
            window_size = 50
            rolling_std = []
            for i in range(window_size, len(self.episode_rewards) + 1):
                window_std = np.std(self.episode_rewards[i-window_size:i])
                rolling_std.append(window_std)
            
            ax5.plot(range(window_size, len(self.episode_rewards) + 1), rolling_std, 
                    color='brown', alpha=0.8, label=f'滑动标准差({window_size})')
            ax5.axhline(y=np.std(self.episode_rewards) * 0.3, color='red', 
                       linestyle='--', alpha=0.7, label='收敛阈值')
        ax5.set_title('收敛稳定性分析')
        ax5.set_xlabel('训练轮次')
        ax5.set_ylabel('标准差')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. 训练统计摘要
        ax6 = axes[2, 1]
        ax6.axis('off')
        
        # 获取收敛指标
        convergence_metrics = self.get_convergence_metrics()
        
        # 创建统计文本
        stats_text = f"""训练统计摘要:
        
总训练轮次: {final_episode}
最终奖励: {convergence_metrics.get('final_reward', 0):.2f}
最大奖励: {convergence_metrics.get('max_reward', 0):.2f}
平均奖励: {convergence_metrics.get('mean_reward', 0):.2f}
奖励改进: {convergence_metrics.get('reward_improvement', 0):.2f}

最终完成率: {convergence_metrics.get('final_completion_rate', 0):.3f}
平均完成率: {convergence_metrics.get('mean_completion_rate', 0):.3f}
完成率改进: {convergence_metrics.get('completion_improvement', 0):.3f}

收敛状态: {convergence_metrics.get('convergence_status', 'unknown')}
稳定性比率: {convergence_metrics.get('stability_ratio', 0):.3f}
改进比率: {convergence_metrics.get('improvement_ratio', 0):.3f}

最终探索率: {self.epsilon:.4f}
内存使用: {len(self.memory) if hasattr(self, 'memory') else 0}
        """
        
        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # 保存图表
        convergence_path = model_save_path.replace('.pth', '_enhanced_convergence.png')
        plt.savefig(convergence_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"增强收敛分析图已保存至: {convergence_path}")
        
        # 生成详细的收敛性报告
        self.generate_convergence_report(model_save_path, convergence_metrics, final_episode)
    
    def generate_convergence_report(self, model_save_path, metrics, final_episode):
        """生成详细的收敛性文字报告"""
        report_path = model_save_path.replace('.pth', '_convergence_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"{self.network_type} 网络收敛性分析报告\n")
            f.write("=" * 60 + "\n\n")
            
            # 基本训练信息
            f.write("训练基本信息:\n")
            f.write("-" * 30 + "\n")
            f.write(f"网络类型: {self.network_type}\n")
            f.write(f"训练轮次: {final_episode}\n")
            f.write(f"设备: {self.device}\n")
            f.write(f"参数数量: {sum(p.numel() for p in self.policy_net.parameters()):,}\n\n")
            
            # 收敛性分析
            f.write("收敛性分析:\n")
            f.write("-" * 30 + "\n")
            f.write(f"收敛状态: {metrics.get('convergence_status', 'unknown')}\n")
            f.write(f"稳定性比率: {metrics.get('stability_ratio', 0):.4f} (< 0.3 为收敛)\n")
            f.write(f"改进比率: {metrics.get('improvement_ratio', 0):.4f}\n\n")
            
            # 奖励分析
            f.write("奖励分析:\n")
            f.write("-" * 30 + "\n")
            f.write(f"最终奖励: {metrics.get('final_reward', 0):.2f}\n")
            f.write(f"最大奖励: {metrics.get('max_reward', 0):.2f}\n")
            f.write(f"平均奖励: {metrics.get('mean_reward', 0):.2f}\n")
            f.write(f"奖励标准差: {metrics.get('reward_std', 0):.2f}\n")
            f.write(f"奖励改进: {metrics.get('reward_improvement', 0):.2f}\n")
            
            if 'recent_mean' in metrics:
                f.write(f"最近50轮平均: {metrics['recent_mean']:.2f}\n")
                f.write(f"最近50轮标准差: {metrics['recent_std']:.2f}\n")
                f.write(f"早期50轮平均: {metrics['early_mean']:.2f}\n")
            f.write("\n")
            
            # 完成率分析
            if 'final_completion_rate' in metrics:
                f.write("完成率分析:\n")
                f.write("-" * 30 + "\n")
                f.write(f"最终完成率: {metrics['final_completion_rate']:.3f}\n")
                f.write(f"最大完成率: {metrics['max_completion_rate']:.3f}\n")
                f.write(f"平均完成率: {metrics['mean_completion_rate']:.3f}\n")
                f.write(f"完成率标准差: {metrics['completion_rate_std']:.3f}\n")
                f.write(f"完成率改进: {metrics['completion_improvement']:.3f}\n")
                
                if 'recent_completion_mean' in metrics:
                    f.write(f"最近50轮完成率: {metrics['recent_completion_mean']:.3f}\n")
                    f.write(f"完成率稳定性: {metrics['completion_stability']:.3f}\n")
                f.write("\n")
            
            # 训练建议
            f.write("训练建议:\n")
            f.write("-" * 30 + "\n")
            
            convergence_status = metrics.get('convergence_status', 'unknown')
            if convergence_status == 'converged':
                f.write("✓ 训练已收敛，模型性能稳定\n")
            elif convergence_status == 'partially_converged':
                f.write("⚠ 部分收敛，建议继续训练或调整超参数\n")
            else:
                f.write("✗ 训练不稳定，建议:\n")
                f.write("  - 降低学习率\n")
                f.write("  - 增加训练轮次\n")
                f.write("  - 调整网络结构\n")
            
            stability_ratio = metrics.get('stability_ratio', 1.0)
            if stability_ratio > 0.6:
                f.write("⚠ 奖励波动较大，建议:\n")
                f.write("  - 使用更平滑的探索策略\n")
                f.write("  - 增加批次大小\n")
                f.write("  - 添加奖励平滑机制\n")
            
            final_completion = metrics.get('final_completion_rate', 0)
            if final_completion < 0.8:
                f.write("⚠ 完成率较低，建议:\n")
                f.write("  - 调整奖励函数\n")
                f.write("  - 增加网络容量\n")
                f.write("  - 优化环境设计\n")
        
        print(f"收敛性报告已保存至: {report_path}")

    def save_model(self, path):
        """保存模型"""
        torch.save(self.policy_net.state_dict(), path)
    
    def load_model(self, path):
        """加载模型"""
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
    
    def get_task_assignments(self):
        """获取任务分配"""
        self.policy_net.eval()
        state = self.env.reset()
        assignments = {u.id: [] for u in self.env.uavs}
        done, step = False, 0
        
        while not done and step < len(self.env.targets) * len(self.env.uavs):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            action_idx = q_values.max(1)[1].item()
            action = self._index_to_action(action_idx)
            target_id, uav_id, _ = action
            
            uav = self.env.uavs[uav_id - 1]
            target = self.env.targets[target_id - 1]
            
            if np.all(uav.resources <= 0):
                step += 1
                continue
            
            assignments[uav_id].append((target_id, action[2]))
            state, _, done, _, _ = self.env.step(action_idx)
            step += 1
        
        return assignments

# =============================================================================
# section 5: 核心功能函数
# =============================================================================
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
                # print(f"警告: UAV {uav_id} 被分配到已满足的目标 {target_id}，跳过此分配")  # 简化输出
                continue
            
            # 检查无人机是否还有资源
            if not np.any(uav_resources[uav_id] > 1e-6):
                # print(f"警告: UAV {uav_id} 资源已耗尽，跳过后续分配")  # 简化输出
                break
            
            # 计算实际贡献
            contribution = np.minimum(uav_resources[uav_id], target_needs[target_id])
            
            # 只有当有实际贡献时才保留此分配
            if np.any(contribution > 1e-6):
                calibrated_assignments[uav_id].append((target_id, phi_idx))
                uav_resources[uav_id] -= contribution
                target_needs[target_id] -= contribution
                # print(f"UAV {uav_id} -> 目标 {target_id}: 贡献 {contribution}")  # 简化输出
            else:
                # print(f"警告: UAV {uav_id} 对目标 {target_id} 无有效贡献，跳过此分配")  # 简化输出
                pass
    
    # 统计校准结果
    original_count = sum(len(tasks) for tasks in task_assignments.values())
    calibrated_count = sum(len(tasks) for tasks in calibrated_assignments.values())
    removed_count = original_count - calibrated_count
    
    print(f"资源分配校准完成:")
    print(f"  原始分配数量: {original_count}")
    print(f"  校准后数量: {calibrated_count}")
    print(f"  移除无效分配: {removed_count}")
    
    return calibrated_assignments

def simple_evaluate_plan(task_assignments, uavs, targets, deadlocked_tasks=None):
    """修复后的计划评估函数 - 基于实际任务完成情况和资源分配"""
    if not task_assignments or not any(task_assignments.values()):
        return {
            'completion_rate': 0.0,
            'satisfied_targets_rate': 0.0,
            'total_reward_score': 0.0,
            'marginal_utility_score': 0.0,
            'resource_efficiency_score': 0.0,
            'distance_cost_score': 0.0,
            'actual_completion_rate': 0.0,
            'resource_satisfaction_rate': 0.0
        }
    
    # 重置目标资源状态以计算实际完成情况
    temp_targets = []
    for target in targets:
        temp_target = type('TempTarget', (), {'id': target.id, 'position': target.position, 
                                           'resources': target.resources.astype(float).copy(), 
                                           'remaining_resources': target.resources.astype(float).copy()})()
        temp_targets.append(temp_target)
    
    # 计算实际资源贡献
    total_contribution = 0.0
    total_available = 0.0
    total_distance = 0.0
    
    for uav in uavs:
        total_available += np.sum(uav.resources)
        uav_remaining = uav.initial_resources.copy().astype(float)
        
        if uav.id in task_assignments:
            for task in task_assignments[uav.id]:
                # 处理不同格式的任务数据
                if isinstance(task, tuple):
                    target_id = task[0]
                    phi_idx = task[1]
                    start_pos = uav.current_position
                else:
                    target_id = task['target_id']
                    start_pos = task['start_pos']
                
                target = next(t for t in temp_targets if t.id == target_id)
                
                # 计算实际贡献
                contribution = np.minimum(target.remaining_resources, uav_remaining)
                actual_contribution = np.sum(contribution)
                
                if actual_contribution > 0:
                    target.remaining_resources = target.remaining_resources - contribution
                    uav_remaining = uav_remaining - contribution
                    total_contribution += actual_contribution
                
                # 计算距离
                distance = np.linalg.norm(start_pos - target.position)
                total_distance += distance
    
    # 计算实际完成率（优化版本，综合考虑多个指标）
    total_targets = len(targets)
    if total_targets > 0:
        # 1. 目标满足数量比例
        completed_targets = sum(1 for t in temp_targets if np.all(t.remaining_resources <= 0))
        target_satisfaction_rate = completed_targets / total_targets
        
        # 2. 总体资源满足率
        total_demand = sum(np.sum(t.resources) for t in targets)
        total_remaining = sum(np.sum(t.remaining_resources) for t in temp_targets)
        resource_satisfaction_rate = (total_demand - total_remaining) / total_demand if total_demand > 0 else 0
        
        # 3. 资源利用率
        total_initial_supply = sum(np.sum(uav.initial_resources) for uav in uavs)
        total_used = total_contribution
        resource_utilization_rate = total_used / total_initial_supply if total_initial_supply > 0 else 0
        
        # 综合完成率 = 目标满足率(50%) + 资源满足率(30%) + 资源利用率(20%)
        actual_completion_rate = (target_satisfaction_rate * 0.5 + 
                                resource_satisfaction_rate * 0.3 + 
                                resource_utilization_rate * 0.2)
    else:
        actual_completion_rate = 0.0
    
    # 计算资源满足率（基于实际贡献）
    total_required = sum(np.sum(t.resources) for t in targets)
    resource_satisfaction_rate = total_contribution / total_required if total_required > 0 else 0.0
    
    # 计算目标满足率（部分或完全满足）
    satisfied_targets = sum(1 for t in temp_targets if np.any(t.resources > t.remaining_resources))
    satisfied_targets_rate = satisfied_targets / total_targets if total_targets > 0 else 0.0
    
    # 计算资源效率
    resource_efficiency_score = 0.0
    if total_available > 0:
        resource_efficiency_score = (total_contribution / total_available) * 500
    
    # 计算距离成本
    distance_cost_score = -total_distance * 0.1
    
    # 计算边际效用
    marginal_utility_score = 0.0
    for target in temp_targets:
        target_initial_total = np.sum(target.resources)
        target_remaining = np.sum(target.remaining_resources)
        if target_initial_total > 0:
            completion_ratio = 1.0 - (target_remaining / target_initial_total)
            # 使用改进的边际效用计算
            marginal_utility = completion_ratio * (1.0 - completion_ratio)
            marginal_utility_score += marginal_utility * 300
    
    # 计算总奖励分数（与优化后的奖励函数保持一致）
    target_completion_score = actual_completion_rate * 500  # 与新的target_completion_reward一致
    completion_bonus = 1000 if actual_completion_rate >= 0.95 else 0  # 放宽完成标准
    
    total_reward_score = (target_completion_score + marginal_utility_score + 
                         resource_efficiency_score + distance_cost_score + completion_bonus)
    
    return {
        'completion_rate': actual_completion_rate,  # 综合完成率
        'satisfied_targets_rate': satisfied_targets_rate,  # 目标满足率
        'target_satisfaction_rate': target_satisfaction_rate,  # 完全满足的目标比例
        'resource_satisfaction_rate': resource_satisfaction_rate,  # 总体资源满足率
        'resource_utilization_rate': resource_utilization_rate,  # 资源利用率
        'total_reward_score': total_reward_score,
        'marginal_utility_score': marginal_utility_score,
        'resource_efficiency_score': resource_efficiency_score,
        'distance_cost_score': distance_cost_score,
        'target_completion_score': target_completion_score,
        'completion_bonus': completion_bonus,
        # 'actual_completion_rate': actual_completion_rate,  # 移除重复定义
        'completed_targets_count': completed_targets,  # 完全满足的目标数量
        'total_targets_count': total_targets,  # 总目标数量
        'total_contribution': total_contribution,  # 总资源贡献
        'total_demand': sum(np.sum(t.resources) for t in targets),  # 总资源需求
        'total_initial_supply': sum(np.sum(uav.initial_resources) for uav in uavs)  # 总初始资源供给
    }

def calculate_economic_sync_speeds(task_assignments, uavs, targets, graph, obstacles, config) -> Tuple[defaultdict, dict]:
    """(已更新) 计算经济同步速度，并返回未完成的任务以进行死锁检测。"""
    # 转换任务数据结构并补充资源消耗
    final_plan = defaultdict(list)
    uav_status = {u.id: {'pos': u.position, 'free_at': 0.0, 'heading': u.heading} for u in uavs}
    remaining_tasks = {uav_id: list(tasks) for uav_id, tasks in task_assignments.items()}
    task_step_counter = defaultdict(lambda: 1)
    
    def _plan_single_leg(args):
        uav_id, start_pos, target_pos, start_heading, end_heading, obstacles, config = args
        planner = PHCurveRRTPlanner(start_pos, target_pos, start_heading, end_heading, obstacles, config)
        return uav_id, planner.plan()
    
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
                uav_id = uav_info['uav_id']
                phi_angle = uav_info['phi_idx'] * (2 * np.pi / config.GRAPH_N_PHI)
                args = (uav_id, uav_status[uav_id]['pos'], target.position, uav_status[uav_id]['heading'], phi_angle, obstacles, config)
                _, plan_result = _plan_single_leg(args)
                if plan_result: path_planners[uav_id] = {'path_points': plan_result[0], 'distance': plan_result[1]}
            
            time_windows = []
            for uav_info in uav_infos:
                uav_id = uav_info['uav_id']
                if uav_id not in path_planners: continue
                uav = next((u for u in uavs if u.id == uav_id), None)
                if not uav: continue
                
                distance = path_planners[uav_id]['distance']
                free_at = uav_status[uav_id]['free_at']
                t_min = free_at + (distance / uav.velocity_range[1])
                t_max = free_at + (distance / uav.velocity_range[0]) if uav.velocity_range[0] > 0 else float('inf')
                t_econ = free_at + (distance / uav.economic_speed)
                time_windows.append({'uav_id': uav_id, 'phi_idx': uav_info['phi_idx'], 't_min': t_min, 't_max': t_max, 't_econ': t_econ})
            
            if not time_windows: continue
            sync_start = max(tw['t_min'] for tw in time_windows)
            sync_end = min(tw['t_max'] for tw in time_windows)
            is_feasible = sync_start <= sync_end + 1e-6
            final_sync_time = np.clip(np.median([tw['t_econ'] for tw in time_windows]), sync_start, sync_end) if is_feasible else sync_start
            group_arrival_times.append({'target_id': target_id, 'arrival_time': final_sync_time, 'uav_infos': time_windows, 'is_feasible': is_feasible, 'path_planners': path_planners})
        
        if not group_arrival_times: break
        next_event = min(group_arrival_times, key=lambda x: x['arrival_time'])
        target_pos = next(t.position for t in targets if t.id == next_event['target_id'])
        
        for uav_info in next_event['uav_infos']:
            uav_id = uav_info['uav_id']
            if uav_id not in next_event['path_planners']: continue
            
            uav, plan_data = next(u for u in uavs if u.id == uav_id), next_event['path_planners'][uav_id]
            travel_time = next_event['arrival_time'] - uav_status[uav_id]['free_at']
            speed = (plan_data['distance'] / travel_time) if travel_time > 1e-6 else uav.velocity_range[1]
            
            final_plan[uav_id].append({
                'target_id': next_event['target_id'],
                'start_pos': uav_status[uav_id]['pos'],
                'speed': np.clip(speed, uav.velocity_range[0], uav.velocity_range[1]),
                'arrival_time': next_event['arrival_time'],
                'step': task_step_counter[uav_id],
                'is_sync_feasible': next_event['is_feasible'],
                'phi_idx': uav_info['phi_idx'],
                'path_points': plan_data['path_points'],
                'distance': plan_data['distance']
            })
            
            task_step_counter[uav_id] += 1
            phi_angle = uav_info['phi_idx'] * (2 * np.pi / config.GRAPH_N_PHI)
            uav_status[uav_id].update(pos=target_pos, free_at=next_event['arrival_time'], heading=phi_angle)
            if remaining_tasks.get(uav_id): remaining_tasks[uav_id].pop(0)
    
    # 应用协同贪婪资源分配策略
    temp_uav_resources = {u.id: u.initial_resources.copy().astype(float) for u in uavs}
    temp_target_resources = {t.id: t.resources.copy().astype(float) for t in targets}
    
    events = defaultdict(list)
    for uav_id, tasks in final_plan.items():
        for task in tasks:
            event_key = (task['arrival_time'], task['target_id'])
            events[event_key].append({'uav_id': uav_id, 'task_ref': task})
    
    for event_key in sorted(events.keys()):
        arrival_time, target_id = event_key
        collaborating_tasks = events[event_key]
        target_remaining = temp_target_resources[target_id].copy()
        
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

def visualize_task_assignments(final_plan, uavs, targets, obstacles, config, scenario_name, 
                             training_time, plan_generation_time, save_plot=True, show_plot=False, 
                             save_report=False, deadlocked_tasks=None, evaluation_metrics=None, output_dir=None):
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

    # --- 后续的可视化和报告生成逻辑将使用上面计算出的精确 resource_cost ---
    # 设置中文字体
    set_chinese_font()
    
    fig, ax = plt.subplots(figsize=(22, 14)); ax.set_facecolor("#f0f0f0");
    for obs in obstacles: 
        if hasattr(obs, 'draw'):
            obs.draw(ax)
        elif hasattr(obs, 'center') and hasattr(obs, 'radius'):
            circle = plt.Circle(obs.center, obs.radius, color='gray', alpha=0.3)
            ax.add_patch(circle)
        elif hasattr(obs, 'vertices'):
            polygon = plt.Polygon(np.array(obs.vertices), color='gray', alpha=0.3)
            ax.add_patch(polygon)

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
        if np.all(total_contribution >= t.resources - 1e-5):
            satisfaction_str, bbox_color = "[OK] 需求满足", 'lightgreen'
        else:
            satisfaction_str, bbox_color = "[NG] 资源不足", 'mistyrose'
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
                    report_header += f"  - {key}: {value:.4f} (归一化: {norm_value:.4f})\n"
                else:
                    report_header += f"  - {key}: {value:.4f} (归一化: {norm_value})\n"
            elif isinstance(value, float):
                report_header += f"  - {key}: {value:.4f}\n"
            else:
                report_header += f"  - {key}: {value}\n"
        report_header += ("-"*30) + "\n\n"
    
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
        report_body_image += "\n"; report_body_file += "\n"
    
    final_report_for_image = report_header + report_body_image; final_report_for_file = report_header + report_body_file
    plt.subplots_adjust(right=0.75); fig.text(0.77, 0.95, final_report_for_image, transform=plt.gcf().transFigure, ha="left", va="top", fontsize=9, bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', ec='grey', alpha=0.9))
    
    train_mode_str = '高精度' if config.USE_PHRRT_DURING_TRAINING else '快速近似'
    
    # 处理training_time格式化问题
    if isinstance(training_time, (tuple, list)):
        actual_episodes = len(training_time[0]) if training_time and len(training_time) > 0 else 0
        estimated_time = actual_episodes * 0.13  # 基于观察到的每轮约0.13秒
        training_time_str = f"{estimated_time:.2f}s ({actual_episodes}轮)"
    else:
        training_time_str = f"{training_time:.2f}s"
    
    title_text = (
        f"多无人机任务分配与路径规划 - {scenario_name}\n"
        f"UAV: {len(uavs)}, 目标: {len(targets)}, 障碍: {len(obstacles)} | 模式: {train_mode_str}\n"
        f"模型训练耗时: {training_time_str} | 方案生成耗时: {plan_generation_time:.2f}s"
    )
    ax.set_title(title_text, fontsize=12, fontweight='bold', pad=20)

    ax.set_xlabel("X坐标 (m)", fontsize=14); ax.set_ylabel("Y坐标 (m)", fontsize=14); ax.legend(loc="lower left"); ax.grid(True, linestyle='--', alpha=0.5, zorder=0); ax.set_aspect('equal', adjustable='box')
    
    if save_plot:
        if output_dir:
            save_path = f'{output_dir}/{scenario_name}_task_assignments.png'
        else:
            output_dir = "output/images"
            os.makedirs(output_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            clean_scenario_name = scenario_name.replace(' ', '_').replace(':', '')
            base_filename = f"{clean_scenario_name}_{timestamp}"
            save_path = os.path.join(output_dir, f"{base_filename}.png")
        plt.savefig(save_path, dpi=300)
        print(f"结果图已保存至: {save_path}")
        
        if save_report:
            # 直接保存到主目录
            report_filepath = f'{output_dir}/{scenario_name}_report.txt'
            try:
                with open(report_filepath, 'w', encoding='utf-8') as f:
                    f.write(final_report_for_file)
                print(f"任务分配报告已保存至: {report_filepath}")
            except Exception as e:
                print(f"错误：无法保存任务报告至 {report_filepath}. 原因: {e}")
                
    if show_plot:
        plt.show()
    plt.close(fig) # 确保无论是否显示，图形对象都被关闭以释放内存

def run_scenario(config, base_uavs, base_targets, obstacles, scenario_name, 
                network_type="SimpleNetwork", save_visualization=True, show_visualization=True, 
                save_report=False, force_retrain=False, incremental_training=False, output_base_dir=None):
    """
    运行场景 - 核心执行器 (优化版本，统一目录结构，支持TensorBoard)
    
    Args:
        config: 配置对象
        base_uavs: UAV列表
        base_targets: 目标列表
        obstacles: 障碍物列表
        scenario_name: 场景名称
        network_type: 网络类型 ("SimpleNetwork", "DeepFCN", "GAT", "DeepFCNResidual")
        save_visualization: 是否保存可视化
        show_visualization: 是否显示可视化
        save_report: 是否保存报告
        force_retrain: 是否强制重新训练
        incremental_training: 是否增量训练
        output_base_dir: 基础输出目录，如果为None则使用默认
        
    Returns:
        final_plan: 最终计划
        training_time: 训练时间
        training_history: 训练历史
        evaluation_metrics: 评估指标
    """
    import time
    import os
    import pickle
    import json
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    network_type = config.NETWORK_TYPE

    # 优化的目录结构 - 统一到一个文件夹
    if output_base_dir:
        output_dir = f"{output_base_dir}/{scenario_name}_{network_type}"
    else:
        output_dir = f"output/{scenario_name}_{network_type}_{timestamp}"
    
    # 创建统一的输出目录结构
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"运行场景: {scenario_name} ({len(base_uavs)}UAV, {len(base_targets)}目标, {len(obstacles)}障碍)")
    print(f"输出目录: {output_dir}")
    
    # 创建图
    graph = DirectedGraph(base_uavs, base_targets, config.GRAPH_N_PHI, obstacles, config)
    
    # 创建求解器 - 动态计算输入维度，支持TensorBoard
    test_env = UAVTaskEnv(base_uavs, base_targets, graph, obstacles, config)
    test_state = test_env.reset()
    input_dim = len(test_state)
    output_dim = test_env.n_actions
    
    # TensorBoard目录 - 直接使用主输出目录
    tensorboard_dir = output_dir
    
    # 使用更深的网络结构
    solver = GraphRLSolver(base_uavs, base_targets, graph, obstacles, 
                          i_dim=input_dim, h_dim=[512, 256, 128, 64], o_dim=output_dim, 
                          config=config, network_type=network_type, 
                          tensorboard_dir=tensorboard_dir)
    
    # 训练模型
    model_save_path = f'{output_dir}/{network_type}_best_model.pth'
    print(f"开始训练 - 网络: {network_type}, 训练轮数: {config.training_config.episodes}")
    # print(f"模型将保存至: {model_save_path}")
    # print(f"TensorBoard日志目录: {tensorboard_dir}")
    # print(f"训练参数: episodes={config.training_config.episodes}, learning_rate={config.training_config.learning_rate}, patience={config.training_config.patience}")
    # print(f"可以使用以下命令查看TensorBoard: tensorboard --logdir={tensorboard_dir}")
    
    training_time = solver.train(
        episodes=config.training_config.episodes, 
        patience=config.training_config.patience, 
        log_interval=config.training_config.log_interval, 
        model_save_path=model_save_path
    )
    
    # 获取任务分配
    print("获取训练后的任务分配...")
    task_assignments = solver.get_task_assignments()
    
    # 校准资源分配
    print("校准资源分配...")
    calibrated_assignments = calibrate_resource_assignments(task_assignments, base_uavs, base_targets)
    
    # 计算路径规划
    print("计算路径规划...")
    final_plan, remaining_tasks = calculate_economic_sync_speeds(
        calibrated_assignments, base_uavs, base_targets, graph, obstacles, config
    )
    
    # 检测真正的死锁任务 - 修复死锁检测逻辑
    deadlocked_tasks = {}
    for uav_id, tasks in remaining_tasks.items():
        if tasks:  # 只有当任务列表非空时才认为是死锁
            deadlocked_tasks[uav_id] = tasks
    
    # 评估解质量
    evaluation_metrics = simple_evaluate_plan(task_assignments, base_uavs, base_targets, deadlocked_tasks)
    
    # 保存训练历史 - 增强版本
    training_history = {
        "episode_rewards": getattr(solver, 'episode_rewards', []),
        "episode_losses": getattr(solver, 'episode_losses', []),
        "epsilon_values": getattr(solver, 'epsilon_values', []),
        "completion_rates": getattr(solver, 'completion_rates', []),
        "episode_steps": getattr(solver, 'episode_steps', []),
        "memory_usage": getattr(solver, 'memory_usage', []),
        "network_type": network_type,
        "scenario_name": scenario_name,
        "training_time": training_time,
        "config_summary": {
            "episodes": config.training_config.episodes,
            "learning_rate": config.training_config.learning_rate,
            "batch_size": config.BATCH_SIZE,
            "gamma": config.GAMMA,
            "epsilon_start": config.training_config.epsilon_start,
            "epsilon_end": config.training_config.epsilon_end,
            "epsilon_decay": config.training_config.epsilon_decay
        }
    }
    
    history_path = f'{output_dir}/training_history.pkl'
    with open(history_path, 'wb') as f:
        pickle.dump(training_history, f)
    
    # 生成训练曲线
    generate_training_curves(training_history, network_type, scenario_name, output_dir)
    
    # 输出解方案信息
    print(f"\n=== {scenario_name} 解方案信息 ===")
    print(f"网络类型: {network_type}")
    print(f"训练时间: {training_time:.2f}秒")
    print(f"任务分配数量: {len(task_assignments)}")
    print(f"死锁任务数量: {len(deadlocked_tasks) if deadlocked_tasks else 0}")
    
    # 统计任务分配详情
    total_assignments = sum(len(assignments) for assignments in task_assignments.values())
    print(f"总任务分配数: {total_assignments}")
    
    # 输出评估指标
    if evaluation_metrics:
        print(f"\n方案评估指标:")
        print(f"{'-' * 50}")
        
        # 核心指标
        if 'completion_rate' in evaluation_metrics:
            print(f"  综合完成率: {evaluation_metrics['completion_rate']:.4f}")
        if 'target_satisfaction_rate' in evaluation_metrics:
            completed = evaluation_metrics.get('completed_targets_count', 0)
            total = evaluation_metrics.get('total_targets_count', 0)
            print(f"  目标完全满足率: {evaluation_metrics['target_satisfaction_rate']:.4f} ({completed}/{total})")
        if 'resource_satisfaction_rate' in evaluation_metrics:
            print(f"  资源满足率: {evaluation_metrics['resource_satisfaction_rate']:.4f}")
        if 'resource_utilization_rate' in evaluation_metrics:
            print(f"  资源利用率: {evaluation_metrics['resource_utilization_rate']:.4f}")
        
        # 详细指标
        if 'total_contribution' in evaluation_metrics and 'total_demand' in evaluation_metrics:
            contrib = evaluation_metrics['total_contribution']
            demand = evaluation_metrics['total_demand']
            print(f"  资源贡献: {contrib:.1f}/{demand:.1f} ({contrib/demand*100:.1f}%)")
        
        if 'total_reward_score' in evaluation_metrics:
            print(f"  总奖励分数: {evaluation_metrics['total_reward_score']:.2f}")
        
        print(f"{'-' * 50}")
    
    # 可视化结果
    print("生成可视化结果...")
    visualize_task_assignments(
        final_plan, base_uavs, base_targets, obstacles, config, scenario_name,
        training_time, 0, save_plot=save_visualization, show_plot=show_visualization,
        output_dir=output_dir
    )
    
    # 生成执行报告
    generate_execution_report(scenario_name, network_type, training_time, 
                            task_assignments, evaluation_metrics, training_history, output_dir)
    
    print(f"场景 {scenario_name} 运行完成")
    print(f"所有输出已保存至: {output_dir}")
    
    return final_plan, training_time, training_history, evaluation_metrics

def main():
    """主函数 - 优化版本，支持统一目录管理"""
    print("多无人机协同任务分配与路径规划系统 - 增强版")
    print("=" * 60)
    
    # 设置中文字体
    set_chinese_font()
    
    # 创建统一的输出目录结构
    os.makedirs('output', exist_ok=True)
    os.makedirs('output/experiments', exist_ok=True)
    os.makedirs('output/temp', exist_ok=True)  # 临时文件
    os.makedirs('output/archive', exist_ok=True)  # 归档文件
    
    # 清理旧的临时文件
    cleanup_temp_files()
    
    # 运行测试场景
    print("加载场景数据...") 
    # uavs, targets, obstacles = get_strategic_trap_scenario(50.0)
    uavs, targets, obstacles = get_new_experimental_scenario(50.0) 
    # uavs, targets, obstacles = get_complex_scenario_v4(30)
    # uavs, targets, obstacles = get_balanced_scenario(50)
    
    
    # 移除场景加载完成的详细输出，保持简洁
    # print(f"场景加载完成:")
    # print(f"  - UAV数量: {len(uavs)}")
    # print(f"  - 目标数量: {len(targets)}")
    # print(f"  - 障碍物数量: {len(obstacles)}")
    
    # 设置优化的训练参数
    config.BATCH_SIZE = 128  # 增大批次大小
    config.LEARNING_RATE = 5e-4  # 降低学习率
    config.GAMMA = 0.99  # 提高折扣因子
    config.MEMORY_CAPACITY = 20000  # 增大经验回放缓冲区
    config.TARGET_UPDATE_FREQ = 10  # 更频繁地更新目标网络
    config.training_config.episodes = 2000  # 增加训练轮数
    config.NETWORK_TYPE = "DeepFCNResidual" # 使用深度残差网络结构 # 网络结构类型: SimpleNetwork、DeepFCN、GAT、DeepFCNResidual
    
    # 运行场景
    final_plan, training_time, training_history, evaluation_metrics = run_scenario(
        config, uavs, targets, obstacles, "试验场景场景",        
        save_visualization=True, show_visualization=False
    )
    
    # 输出最终结果
    print(f"\n{'='*60}")
    print(f"系统运行完成")
    print(f"{'='*60}")
    # print(f"网络架构: {config.NETWORK_TYPE}")
    print(f"训练时间: {training_time:.2f}秒")
    print(f"参与UAV: {len(final_plan) if final_plan else 0}个")
    
    # 输出核心性能指标
    # if evaluation_metrics:
    #     print(f"\n核心性能指标:")
    #     print(f"{'-' * 30}")
    #     if 'completion_rate' in evaluation_metrics:
    #         print(f"综合完成率: {evaluation_metrics['completion_rate']:.4f}")
    #     if 'target_satisfaction_rate' in evaluation_metrics:
    #         completed = evaluation_metrics.get('completed_targets_count', 0)
    #         total = evaluation_metrics.get('total_targets_count', 0)
    #         print(f"目标完成: {completed}/{total} ({evaluation_metrics['target_satisfaction_rate']:.1%})")
    #     if 'resource_utilization_rate' in evaluation_metrics:
    #         print(f"资源利用率: {evaluation_metrics['resource_utilization_rate']:.1%}")
    #     print(f"{'-' * 30}")
    
    # 生成训练报告
    report_path = f'output/reports/{time.strftime("%Y%m%d-%H%M%S")}_training_report.txt'
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"训练报告 - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n")        
        f.write(f"训练时间: {training_time:.2f}秒\n")
        f.write(f"UAV数量: {len(uavs)}\n")
        f.write(f"目标数量: {len(targets)}\n")
        f.write(f"障碍物数量: {len(obstacles)}\n")
        f.write(f"任务分配数量: {len(final_plan) if final_plan else 0}\n")
        f.write(f"目标满足率: {evaluation_metrics.get('satisfied_targets_rate', 0):.4f}\n")
        f.write(f"资源利用率: {evaluation_metrics.get('resource_utilization_rate', 0):.4f}\n")
    
    print(f"训练报告已保存至: {report_path}")
    print("系统运行完成！")

def cleanup_temp_files():
    """清理临时文件"""
    import shutil
    import glob
    
    print("清理临时文件...")
    
    # 移动旧的输出文件到temp目录
    old_patterns = [
        'output/*.png',
        'output/*.txt', 
        'output/*.pkl',
        'output/*.json'
    ]
    
    temp_dir = 'output/temp'
    os.makedirs(temp_dir, exist_ok=True)
    
    moved_count = 0
    for pattern in old_patterns:
        for file_path in glob.glob(pattern):
            if os.path.isfile(file_path):
                filename = os.path.basename(file_path)
                new_path = os.path.join(temp_dir, filename)
                try:
                    shutil.move(file_path, new_path)
                    moved_count += 1
                except Exception as e:
                    print(f"移动文件失败 {file_path}: {e}")
    
    if moved_count > 0:
        print(f"已移动 {moved_count} 个临时文件到 {temp_dir}")
    
    # 清理空的旧目录
    old_dirs = ['output/models', 'output/images', 'output/reports', 'output/curves']
    for old_dir in old_dirs:
        if os.path.exists(old_dir) and not os.listdir(old_dir):
            try:
                os.rmdir(old_dir)
                print(f"已删除空目录: {old_dir}")
            except Exception:
                pass

def generate_training_curves(training_history, network_type, scenario_name, output_dir):
    """生成训练曲线 - 增强版本，包含更多调试信息"""
    import matplotlib.pyplot as plt
    
    # 设置中文字体
    set_chinese_font()
    
    # 创建更详细的训练曲线图
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f'{scenario_name} - {network_type} 训练分析', fontsize=16)
    
    # 奖励曲线
    if training_history.get('episode_rewards'):
        rewards = training_history['episode_rewards']
        axes[0, 0].plot(rewards, alpha=0.7, label='原始奖励')
        # 添加移动平均
        if len(rewards) > 10:
            window = min(50, len(rewards) // 10)
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axes[0, 0].plot(range(window-1, len(rewards)), moving_avg, 'r-', linewidth=2, label=f'移动平均({window})')
        axes[0, 0].set_title('奖励曲线')
        axes[0, 0].set_xlabel('回合')
        axes[0, 0].set_ylabel('奖励')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
    
    # 损失曲线
    if training_history.get('episode_losses'):
        losses = training_history['episode_losses']
        axes[0, 1].plot(losses, alpha=0.7, label='原始损失')
        # 添加移动平均
        if len(losses) > 10:
            window = min(50, len(losses) // 10)
            moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
            axes[0, 1].plot(range(window-1, len(losses)), moving_avg, 'r-', linewidth=2, label=f'移动平均({window})')
        axes[0, 1].set_title('损失曲线')
        axes[0, 1].set_xlabel('回合')
        axes[0, 1].set_ylabel('损失')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        axes[0, 1].set_yscale('log')  # 使用对数刻度
    
    # 探索率曲线
    if training_history.get('epsilon_values'):
        axes[1, 0].plot(training_history['epsilon_values'])
        axes[1, 0].set_title('探索率衰减')
        axes[1, 0].set_xlabel('回合')
        axes[1, 0].set_ylabel('探索率 (ε)')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 完成率曲线
    if training_history.get('completion_rates'):
        completion_rates = training_history['completion_rates']
        axes[1, 1].plot(completion_rates, alpha=0.7, label='原始完成率')
        # 添加移动平均
        if len(completion_rates) > 10:
            window = min(50, len(completion_rates) // 10)
            moving_avg = np.convolve(completion_rates, np.ones(window)/window, mode='valid')
            axes[1, 1].plot(range(window-1, len(completion_rates)), moving_avg, 'r-', linewidth=2, label=f'移动平均({window})')
        axes[1, 1].set_title('任务完成率')
        axes[1, 1].set_xlabel('回合')
        axes[1, 1].set_ylabel('完成率')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
    
    # 每回合步数
    if training_history.get('episode_steps'):
        axes[2, 0].plot(training_history['episode_steps'])
        axes[2, 0].set_title('每回合步数')
        axes[2, 0].set_xlabel('回合')
        axes[2, 0].set_ylabel('步数')
        axes[2, 0].grid(True, alpha=0.3)
    
    # 经验回放池使用情况
    if training_history.get('memory_usage'):
        axes[2, 1].plot(training_history['memory_usage'])
        axes[2, 1].set_title('经验回放池使用情况')
        axes[2, 1].set_xlabel('回合')
        axes[2, 1].set_ylabel('存储的经验数量')
        axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = f'{output_dir}/training_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"训练曲线已保存至: {save_path}")
    
    # 生成收敛性分析报告
    generate_convergence_analysis(training_history, network_type, scenario_name, output_dir)

def generate_convergence_analysis(training_history, network_type, scenario_name, output_dir):
    """生成收敛性分析报告"""
    analysis_path = f'{output_dir}/convergence_analysis.txt'
    
    with open(analysis_path, 'w', encoding='utf-8') as f:
        f.write(f"收敛性分析报告\n")
        f.write(f"================\n\n")
        f.write(f"网络类型: {network_type}\n")
        f.write(f"场景名称: {scenario_name}\n\n")
        
        # 奖励分析
        if training_history.get('episode_rewards'):
            rewards = training_history['episode_rewards']
            f.write(f"奖励分析:\n")
            f.write(f"  - 初始奖励: {rewards[0]:.2f}\n")
            f.write(f"  - 最终奖励: {rewards[-1]:.2f}\n")
            f.write(f"  - 最大奖励: {max(rewards):.2f}\n")
            f.write(f"  - 平均奖励: {np.mean(rewards):.2f}\n")
            f.write(f"  - 奖励标准差: {np.std(rewards):.2f}\n")
            
            # 收敛性检查
            if len(rewards) > 100:
                last_100 = rewards[-100:]
                first_100 = rewards[:100]
                improvement = np.mean(last_100) - np.mean(first_100)
                f.write(f"  - 前100回合平均: {np.mean(first_100):.2f}\n")
                f.write(f"  - 后100回合平均: {np.mean(last_100):.2f}\n")
                f.write(f"  - 改进程度: {improvement:.2f}\n")
                
                # 稳定性分析
                last_50_std = np.std(rewards[-50:])
                f.write(f"  - 最后50回合标准差: {last_50_std:.2f}\n")
                if last_50_std < np.std(rewards) * 0.5:
                    f.write(f"  - 收敛状态: 良好 (变化较小)\n")
                else:
                    f.write(f"  - 收敛状态: 不稳定 (仍在波动)\n")
        
        # 损失分析
        if training_history.get('episode_losses'):
            losses = training_history['episode_losses']
            f.write(f"\n损失分析:\n")
            f.write(f"  - 初始损失: {losses[0]:.4f}\n")
            f.write(f"  - 最终损失: {losses[-1]:.4f}\n")
            f.write(f"  - 最小损失: {min(losses):.4f}\n")
            f.write(f"  - 平均损失: {np.mean(losses):.4f}\n")
            
            # 损失趋势
            if len(losses) > 50:
                recent_trend = np.mean(losses[-50:]) - np.mean(losses[-100:-50]) if len(losses) > 100 else 0
                f.write(f"  - 近期趋势: {'下降' if recent_trend < 0 else '上升'} ({recent_trend:.4f})\n")
        
        # 完成率分析
        if training_history.get('completion_rates'):
            completion_rates = training_history['completion_rates']
            f.write(f"\n完成率分析:\n")
            f.write(f"  - 初始完成率: {completion_rates[0]:.3f}\n")
            f.write(f"  - 最终完成率: {completion_rates[-1]:.3f}\n")
            f.write(f"  - 最高完成率: {max(completion_rates):.3f}\n")
            f.write(f"  - 平均完成率: {np.mean(completion_rates):.3f}\n")
            
            # 达到高完成率的频率
            high_completion_count = sum(1 for rate in completion_rates if rate > 0.8)
            f.write(f"  - 高完成率(>0.8)频率: {high_completion_count}/{len(completion_rates)} ({high_completion_count/len(completion_rates)*100:.1f}%)\n")
        
        # 训练效率分析
        if training_history.get('training_time'):
            f.write(f"\n训练效率:\n")
            f.write(f"  - 总训练时间: {training_history['training_time']:.2f}秒\n")
            f.write(f"  - 每回合平均时间: {training_history['training_time']/len(rewards):.3f}秒\n")
        
        # 收敛建议
        f.write(f"\n收敛性建议:\n")
        if training_history.get('episode_rewards'):
            final_reward = rewards[-1]
            max_reward = max(rewards)
            if final_reward < max_reward * 0.8:
                f.write(f"  - 建议增加训练回合数或调整学习率\n")
            if np.std(rewards[-50:]) > np.std(rewards) * 0.7:
                f.write(f"  - 建议降低学习率以提高稳定性\n")
            if max(completion_rates) < 0.5:
                f.write(f"  - 建议检查奖励函数设计和网络结构\n")
    
    print(f"收敛性分析报告已保存至: {analysis_path}")

def generate_execution_report(scenario_name, network_type, training_time, 
                            task_assignments, evaluation_metrics, training_history, output_dir):
    """生成执行报告 - 增强版本"""
    import json
    
    # 计算更详细的统计信息
    training_stats = {}
    if training_history.get('episode_rewards'):
        rewards = training_history['episode_rewards']
        training_stats.update({
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "reward_max": float(max(rewards)),
            "reward_min": float(min(rewards)),
            "reward_final": float(rewards[-1]),
            "reward_improvement": float(rewards[-1] - rewards[0]) if len(rewards) > 1 else 0
        })
    
    if training_history.get('episode_losses'):
        losses = training_history['episode_losses']
        training_stats.update({
            "loss_mean": float(np.mean(losses)),
            "loss_std": float(np.std(losses)),
            "loss_final": float(losses[-1]),
            "loss_min": float(min(losses))
        })
    
    report = {
        "scenario_name": scenario_name,
        "network_type": network_type,
        "training_time": training_time,
        "task_assignments_count": len(task_assignments),
        "evaluation_metrics": evaluation_metrics,
        "training_statistics": training_stats,
        "training_history_summary": {
            "total_episodes": len(training_history.get('episode_rewards', [])),
            "final_reward": training_history.get('episode_rewards', [0])[-1] if training_history.get('episode_rewards') else 0,
            "final_loss": training_history.get('episode_losses', [0])[-1] if training_history.get('episode_losses') else 0,
            "final_epsilon": training_history.get('epsilon_values', [0])[-1] if training_history.get('epsilon_values') else 0,
            "final_completion_rate": training_history.get('completion_rates', [0])[-1] if training_history.get('completion_rates') else 0
        },
        "config_summary": training_history.get('config_summary', {})
    }
    
    # 转换numpy类型后再序列化
    serializable_report = convert_numpy_types(report)
    with open(f'{output_dir}/execution_report.json', 'w', encoding='utf-8') as f:
        json.dump(serializable_report, f, ensure_ascii=False, indent=2)
    
    # 生成详细的文本报告
    with open(f'{output_dir}/execution_report.txt', 'w', encoding='utf-8') as f:
        f.write(f"场景执行报告\n")
        f.write(f"=============\n\n")
        f.write(f"场景名称: {scenario_name}\n")
        f.write(f"网络类型: {network_type}\n")
        f.write(f"训练时间: {training_time:.2f}秒\n")
        f.write(f"任务分配数量: {len(task_assignments)}\n\n")
        
        if evaluation_metrics:
            f.write(f"评估指标:\n")
            for key, value in evaluation_metrics.items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.4f}\n")
                else:
                    f.write(f"  {key}: {value}\n")
        
        f.write(f"\n训练统计:\n")
        for key, value in training_stats.items():
            f.write(f"  {key}: {value:.4f}\n")
        
        f.write(f"\n训练历史摘要:\n")
        summary = report["training_history_summary"]
        f.write(f"  总回合数: {summary['total_episodes']}\n")
        f.write(f"  最终奖励: {summary['final_reward']:.2f}\n")
        f.write(f"  最终损失: {summary['final_loss']:.4f}\n")
        f.write(f"  最终探索率: {summary['final_epsilon']:.4f}\n")
        f.write(f"  最终完成率: {summary['final_completion_rate']:.4f}\n")
        
        f.write(f"\n配置参数:\n")
        config_summary = report.get("config_summary", {})
        for key, value in config_summary.items():
            f.write(f"  {key}: {value}\n")
    
    print(f"执行报告已保存至: {output_dir}/execution_report.txt")

if __name__ == "__main__":
    main()