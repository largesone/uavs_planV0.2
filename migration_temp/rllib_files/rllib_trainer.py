# -*- coding: utf-8 -*-
# 文件名: rllib_trainer.py
# 描述: 使用Ray RLlib进行UAV任务分配的强化学习训练

import os
import sys
import numpy as np
import time
import pickle
from typing import Dict, Any, Optional
import ray
from ray import tune
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.env.env_context import EnvContext
from ray.tune.registry import register_env
from ray.rllib.utils.metrics import NUM_AGENT_STEPS_SAMPLED, NUM_ENV_STEPS_SAMPLED

# 导入项目模块
from entities import UAV, Target
from path_planning import PHCurveRRTPlanner
from scenarios import get_simple_scenario, get_complex_scenario
from config import Config
from rllib_env import UAVTaskEnvRLlib

def env_creator(env_config: Dict[str, Any]) -> UAVTaskEnvRLlib:
    """
    环境创建函数，用于RLlib注册环境
    
    Args:
        env_config: 环境配置字典
        
    Returns:
        UAVTaskEnvRLlib: 创建的环境实例
    """
    # 从配置中获取场景信息
    scenario_name = env_config.get('scenario_name', 'simple')
    obstacle_tolerance = env_config.get('obstacle_tolerance', 50.0)
    
    # 创建场景
    if scenario_name == 'simple':
        uavs, targets, obstacles = get_simple_scenario(obstacle_tolerance)
    elif scenario_name == 'complex':
        uavs, targets, obstacles = get_complex_scenario(obstacle_tolerance)
    else:
        raise ValueError(f"未知场景: {scenario_name}")
    
    # 创建有向图
    from main import DirectedGraph
    graph = DirectedGraph(uavs, targets, env_config.get('n_phi', 6), obstacles)
    
    # 创建配置
    config = Config()
    
    # 创建环境
    env = UAVTaskEnvRLlib(uavs, targets, graph, obstacles, config)
    
    return env

def create_dqn_config() -> DQNConfig:
    """
    创建DQN配置
    
    Returns:
        DQNConfig: 配置好的DQN算法配置
    """
    config = DQNConfig()
    
    # 基础训练参数
    config = config.training(
        # 网络配置 - 类似OptimizedDeepFCN
        model={
            "fcnet_hiddens": [256, 256, 128],  # 三层全连接网络
            "fcnet_activation": "relu",
            "post_fcnet_hiddens": [64],  # 额外的后处理层
            "post_fcnet_activation": "relu",
            "no_final_linear": False,
            "vf_share_layers": False,
            "free_log_std": False,
            "use_ortho_init": True,
        },
        # 经验回放配置
        replay_buffer_config={
            "type": "MultiAgentReplayBuffer",
            "capacity": 50000,
        },
        # 目标网络更新
        target_network_update_freq=10,
        # 梯度裁剪
        grad_clip=1.0,
        # 学习率
        lr=0.001,
        # 批次大小
        train_batch_size=128,
        # 探索配置
        exploration_config={
            "type": "EpsilonGreedy",
            "initial_epsilon": 1.0,
            "final_epsilon": 0.05,
            "epsilon_timesteps": 10000,
        },
        # 双DQN
        double_q=True,
        # 优先经验回放
        prioritized_replay=True,
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta=0.4,
        prioritized_replay_eps=1e-6,
        # 噪声网络
        noisy=False,
        # 参数噪声
        parameter_noise=False,
        # 训练频率
        train_freq=1,
        # 采样频率
        sample_batch_size=4,
        # 优化器
        optimizer={
            "type": "Adam",
            "lr": 0.001,
            "epsilon": 1e-8,
        },
        # 损失函数
        loss_function="huber",
        # 奖励缩放
        reward_scaling=1.0,
        # 折扣因子
        gamma=0.98,
        # 软更新
        soft_q=False,
    )
    
    # 环境配置
    config = config.environment(
        env="uav_task_env",
        env_config={
            "scenario_name": "simple",
            "obstacle_tolerance": 50.0,
            "n_phi": 6,
        },
        clip_actions=False,
        normalize_actions=False,
        clip_rewards=None,
        normalize_rewards=False,
        observation_filter="NoFilter",
        reward_filter="NoFilter",
    )
    
    # 资源配置
    config = config.resources(
        num_gpus=0,  # 如果有GPU可以设置为1
        num_cpus_per_worker=1,
        num_gpus_per_worker=0,
        num_cpus_for_local_worker=1,
        num_gpus_for_local_worker=0,
    )
    
    # 并行配置
    config = config.rollouts(
        num_rollout_workers=4,  # 并行采样工作进程数
        rollout_fragment_length=200,
        batch_mode="complete_episodes",
        enable_multiple_episodes_in_batch=True,
        enable_rl_module_api=False,
        enable_connectors=False,
    )
    
    # 调试配置
    config = config.debugging(
        log_level="INFO",
        seed=42,
        log_sys_usage=True,
        fake_sampler=False,
    )
    
    # 报告配置
    config = config.reporting(
        keep_per_episode_custom_metrics=True,
        metrics_episode_collection_timeout_s=60,
        metrics_num_episodes_for_smoothing=100,
        min_time_s_per_iteration=0,
        min_train_timesteps_per_iteration=0,
        min_sample_timesteps_per_iteration=0,
    )
    
    return config

def train_with_rllib(scenario_name: str = "simple", 
                    num_episodes: int = 1000,
                    checkpoint_dir: str = "output/rllib_checkpoints"):
    """
    使用RLlib训练UAV任务分配模型
    
    Args:
        scenario_name: 场景名称
        num_episodes: 训练轮次
        checkpoint_dir: 检查点保存目录
    """
    # 初始化Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    # 注册环境
    register_env("uav_task_env", env_creator)
    
    # 创建DQN配置
    config = create_dqn_config()
    
    # 更新环境配置
    config = config.environment(
        env_config={
            "scenario_name": scenario_name,
            "obstacle_tolerance": 50.0,
            "n_phi": 6,
        }
    )
    
    # 创建算法实例
    algo = config.build()
    
    # 训练循环
    print(f"开始训练 {scenario_name} 场景，共 {num_episodes} 轮...")
    start_time = time.time()
    
    for i in range(num_episodes):
        # 训练一步
        result = algo.train()
        
        # 打印训练信息
        if i % 10 == 0:
            episode_reward_mean = result.get("episode_reward_mean", 0)
            episode_len_mean = result.get("episode_len_mean", 0)
            policy_reward_mean = result.get("policy_reward_mean", {})
            
            print(f"Episode {i}/{num_episodes}")
            print(f"  平均奖励: {episode_reward_mean:.2f}")
            print(f"  平均长度: {episode_len_mean:.2f}")
            print(f"  策略奖励: {policy_reward_mean}")
            print(f"  采样步数: {result.get(NUM_ENV_STEPS_SAMPLED, 0)}")
            print(f"  智能体步数: {result.get(NUM_AGENT_STEPS_SAMPLED, 0)}")
            print("-" * 50)
        
        # 保存检查点
        if i % 100 == 0 and i > 0:
            checkpoint_path = algo.save(checkpoint_dir)
            print(f"检查点已保存到: {checkpoint_path}")
    
    training_time = time.time() - start_time
    print(f"训练完成，耗时: {training_time:.2f}秒")
    
    # 保存最终模型
    final_checkpoint = algo.save(checkpoint_dir)
    print(f"最终模型已保存到: {final_checkpoint}")
    
    return algo, final_checkpoint

def evaluate_model(algo, scenario_name: str = "simple", num_episodes: int = 10):
    """
    评估训练好的模型
    
    Args:
        algo: 训练好的算法实例
        scenario_name: 场景名称
        num_episodes: 评估轮次
    """
    print(f"开始评估 {scenario_name} 场景，共 {num_episodes} 轮...")
    
    # 创建环境
    env_config = {
        "scenario_name": scenario_name,
        "obstacle_tolerance": 50.0,
        "n_phi": 6,
    }
    env = env_creator(env_config)
    
    total_rewards = []
    completion_rates = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0
        
        while True:
            # 使用策略网络选择动作
            action = algo.compute_single_action(obs)
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
            if terminated or truncated:
                break
        
        # 计算完成率
        task_assignments = env.get_task_assignments()
        total_tasks = sum(len(assignments) for assignments in task_assignments.values())
        completion_rate = total_tasks / (len(env.targets) * len(env.uavs)) if len(env.targets) * len(env.uavs) > 0 else 0
        
        total_rewards.append(episode_reward)
        completion_rates.append(completion_rate)
        
        print(f"Episode {episode + 1}: 奖励={episode_reward:.2f}, 完成率={completion_rate:.2f}")
    
    avg_reward = np.mean(total_rewards)
    avg_completion_rate = np.mean(completion_rates)
    
    print(f"评估结果:")
    print(f"  平均奖励: {avg_reward:.2f}")
    print(f"  平均完成率: {avg_completion_rate:.2f}")
    print(f"  奖励标准差: {np.std(total_rewards):.2f}")
    print(f"  完成率标准差: {np.std(completion_rates):.2f}")
    
    return {
        "avg_reward": avg_reward,
        "avg_completion_rate": avg_completion_rate,
        "reward_std": np.std(total_rewards),
        "completion_rate_std": np.std(completion_rates),
    }

def main():
    """主函数"""
    print("=== UAV任务分配RLlib训练器 ===")
    
    # 训练参数
    scenario_name = "simple"  # 或 "complex"
    num_episodes = 500
    checkpoint_dir = "output/rllib_checkpoints"
    
    # 创建输出目录
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 训练模型
    algo, checkpoint_path = train_with_rllib(scenario_name, num_episodes, checkpoint_dir)
    
    # 评估模型
    evaluation_results = evaluate_model(algo, scenario_name, num_episodes=10)
    
    # 保存评估结果
    results_file = os.path.join(checkpoint_dir, "evaluation_results.pkl")
    with open(results_file, 'wb') as f:
        pickle.dump(evaluation_results, f)
    
    print(f"评估结果已保存到: {results_file}")
    print("训练和评估完成！")

if __name__ == "__main__":
    main() 