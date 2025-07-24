# -*- coding: utf-8 -*-
# 文件名: main_rllib.py
# 描述: 使用Ray RLlib的UAV任务分配主程序

import os
import sys
import time
import argparse
from typing import Dict, Any

# 导入RLlib相关模块
from rllib_trainer import train_with_rllib, evaluate_model, env_creator
from rllib_config import get_rllib_config
from rllib_env import UAVTaskEnvRLlib

def run_rllib_training(scenario_name: str = "simple", 
                      num_episodes: int = 1000,
                      checkpoint_dir: str = "output/rllib_checkpoints"):
    """
    运行RLlib训练
    
    Args:
        scenario_name: 场景名称
        num_episodes: 训练轮次
        checkpoint_dir: 检查点目录
    """
    print(f"=== 开始RLlib训练 ===")
    print(f"场景: {scenario_name}")
    print(f"训练轮次: {num_episodes}")
    print(f"检查点目录: {checkpoint_dir}")
    print("-" * 50)
    
    # 创建输出目录
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 训练模型
    start_time = time.time()
    algo, checkpoint_path = train_with_rllib(scenario_name, num_episodes, checkpoint_dir)
    training_time = time.time() - start_time
    
    print(f"\n训练完成!")
    print(f"训练耗时: {training_time:.2f}秒")
    print(f"最终检查点: {checkpoint_path}")
    
    return algo, checkpoint_path, training_time

def run_evaluation(algo, scenario_name: str = "simple", num_episodes: int = 10):
    """
    运行模型评估
    
    Args:
        algo: 训练好的算法实例
        scenario_name: 场景名称
        num_episodes: 评估轮次
    """
    print(f"\n=== 开始模型评估 ===")
    print(f"场景: {scenario_name}")
    print(f"评估轮次: {num_episodes}")
    print("-" * 50)
    
    # 评估模型
    evaluation_results = evaluate_model(algo, scenario_name, num_episodes)
    
    print(f"\n评估完成!")
    print(f"平均奖励: {evaluation_results['avg_reward']:.2f}")
    print(f"平均完成率: {evaluation_results['avg_completion_rate']:.2f}")
    
    return evaluation_results

def run_comparison(scenario_name: str = "simple"):
    """
    运行性能对比测试
    
    Args:
        scenario_name: 场景名称
    """
    print(f"\n=== 性能对比测试 ===")
    print(f"场景: {scenario_name}")
    print("-" * 50)
    
    # 获取配置
    config = get_rllib_config(scenario_name)
    
    # 测试不同配置的性能
    configs_to_test = [
        ("基础DQN", {"fcnet_hiddens": [256, 128], "post_fcnet_hiddens": []}),
        ("深度DQN", {"fcnet_hiddens": [512, 256, 128], "post_fcnet_hiddens": [64]}),
        ("宽网络DQN", {"fcnet_hiddens": [1024, 512, 256], "post_fcnet_hiddens": [128, 64]}),
    ]
    
    results = {}
    
    for config_name, network_config in configs_to_test:
        print(f"\n测试配置: {config_name}")
        print(f"网络结构: {network_config}")
        
        # 更新配置
        config.fcnet_hiddens = network_config["fcnet_hiddens"]
        config.post_fcnet_hiddens = network_config["post_fcnet_hiddens"]
        config.num_episodes = 200  # 减少训练轮次用于快速测试
        
        # 训练
        algo, checkpoint_path, training_time = run_rllib_training(
            scenario_name, config.num_episodes, 
            f"output/rllib_checkpoints/{config_name.lower().replace(' ', '_')}"
        )
        
        # 评估
        eval_results = run_evaluation(algo, scenario_name, 5)
        
        # 记录结果
        results[config_name] = {
            "training_time": training_time,
            "checkpoint_path": checkpoint_path,
            "evaluation_results": eval_results,
            "network_config": network_config
        }
    
    # 打印对比结果
    print(f"\n=== 性能对比结果 ===")
    print(f"{'配置':<15} {'训练时间':<10} {'平均奖励':<10} {'完成率':<10}")
    print("-" * 50)
    
    for config_name, result in results.items():
        training_time = result["training_time"]
        avg_reward = result["evaluation_results"]["avg_reward"]
        completion_rate = result["evaluation_results"]["avg_completion_rate"]
        
        print(f"{config_name:<15} {training_time:<10.1f} {avg_reward:<10.2f} {completion_rate:<10.2f}")
    
    return results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="UAV任务分配RLlib训练器")
    parser.add_argument("--scenario", type=str, default="simple", 
                       choices=["simple", "complex"], help="场景名称")
    parser.add_argument("--episodes", type=int, default=1000, help="训练轮次")
    parser.add_argument("--mode", type=str, default="train", 
                       choices=["train", "evaluate", "compare"], help="运行模式")
    parser.add_argument("--checkpoint", type=str, help="检查点路径（用于评估模式）")
    
    args = parser.parse_args()
    
    print("=== UAV任务分配RLlib系统 ===")
    print(f"场景: {args.scenario}")
    print(f"模式: {args.mode}")
    print(f"训练轮次: {args.episodes}")
    print("-" * 50)
    
    if args.mode == "train":
        # 训练模式
        algo, checkpoint_path, training_time = run_rllib_training(
            args.scenario, args.episodes
        )
        
        # 评估训练好的模型
        evaluation_results = run_evaluation(algo, args.scenario)
        
        print(f"\n=== 训练总结 ===")
        print(f"训练耗时: {training_time:.2f}秒")
        print(f"检查点路径: {checkpoint_path}")
        print(f"最终性能: 奖励={evaluation_results['avg_reward']:.2f}, "
              f"完成率={evaluation_results['avg_completion_rate']:.2f}")
    
    elif args.mode == "evaluate":
        # 评估模式
        if not args.checkpoint:
            print("错误: 评估模式需要指定检查点路径 (--checkpoint)")
            return
        
        # 加载模型并评估
        import ray
        from ray.rllib.algorithms.dqn import DQN
        
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        # 创建算法实例并加载检查点
        from rllib_trainer import create_dqn_config
        config = create_dqn_config()
        algo = config.build()
        algo.restore(args.checkpoint)
        
        # 评估
        evaluation_results = run_evaluation(algo, args.scenario)
        
        print(f"\n=== 评估总结 ===")
        print(f"检查点: {args.checkpoint}")
        print(f"性能: 奖励={evaluation_results['avg_reward']:.2f}, "
              f"完成率={evaluation_results['avg_completion_rate']:.2f}")
    
    elif args.mode == "compare":
        # 对比模式
        results = run_comparison(args.scenario)
        
        print(f"\n=== 对比总结 ===")
        best_config = max(results.keys(), 
                         key=lambda k: results[k]["evaluation_results"]["avg_reward"])
        print(f"最佳配置: {best_config}")
        print(f"最佳奖励: {results[best_config]['evaluation_results']['avg_reward']:.2f}")
    
    print("\n程序执行完成!")

if __name__ == "__main__":
    main() 