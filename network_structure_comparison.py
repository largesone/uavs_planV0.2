# -*- coding: utf-8 -*-
"""
网络结构对比测试脚本
测试三种网络结构：DeepFCN、DeepFCN_Residual、GAT
1. 简单场景：测试训练收敛性
2. 战略陷阱场景：测试解质量
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from typing import Dict, List, Tuple
import pandas as pd

from enhanced_rl_solver import EnhancedRLSolver
from scenarios import get_simple_convergence_test_scenario, get_strategic_trap_scenario
from main import DirectedGraph
from config import Config

class NetworkStructureTester:
    """网络结构对比测试器"""
    
    def __init__(self):
        self.config = Config()
        self.network_types = ['DeepFCN', 'DeepFCN_Residual', 'GAT']
        self.results = {}
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
    
    def test_convergence_simple_scenario(self, episodes=300):
        """测试简单场景下的训练收敛性"""
        print("=== 简单场景训练收敛性测试 ===")
        
        # 创建简单场景
        uavs, targets, obstacles = get_simple_convergence_test_scenario()
        graph = DirectedGraph(uavs, targets, self.config.GRAPH_N_PHI, obstacles)
        
        convergence_results = {}
        
        for network_type in self.network_types:
            print(f"\n测试网络结构: {network_type}")
            
            # 创建求解器
            solver = EnhancedRLSolver(uavs, targets, graph, obstacles, self.config, 
                                    network_type=network_type, use_ph_rrt=False)
            
            # 训练
            start_time = time.time()
            history = solver.train(episodes, patience=50, log_interval=20)
            training_time = time.time() - start_time
            
            # 保存结果
            convergence_results[network_type] = {
                'history': history,
                'training_time': training_time,
                'final_avg_reward': np.mean(history['episode_rewards'][-50:]),
                'max_reward': max(history['episode_rewards']),
                'convergence_episode': len(history['episode_rewards']),
                'final_epsilon': history['epsilon_history'][-1],
                'avg_deadlock_events': np.mean(history['deadlock_events']),
                'avg_resource_satisfaction': np.mean(history['resource_satisfaction']),
                'loss_history': history.get('loss_history', [])
            }
            
            print(f"  训练时间: {training_time:.2f}秒")
            print(f"  最终平均奖励: {convergence_results[network_type]['final_avg_reward']:.2f}")
            print(f"  最大奖励: {convergence_results[network_type]['max_reward']:.2f}")
            print(f"  收敛轮次: {convergence_results[network_type]['convergence_episode']}")
            print(f"  平均死锁事件: {convergence_results[network_type]['avg_deadlock_events']:.2f}")
            print(f"  平均资源满足率: {convergence_results[network_type]['avg_resource_satisfaction']:.3f}")
        
        self.results['convergence'] = convergence_results
        return convergence_results
    
    def test_solution_quality_strategic_scenario(self, episodes=200):
        """测试战略陷阱场景下的解质量"""
        print("\n=== 战略陷阱场景解质量测试 ===")
        
        # 创建战略陷阱场景
        uavs, targets, obstacles = get_strategic_trap_scenario()
        graph = DirectedGraph(uavs, targets, self.config.GRAPH_N_PHI, obstacles)
        
        quality_results = {}
        
        for network_type in self.network_types:
            print(f"\n测试网络结构: {network_type}")
            
            # 创建求解器
            solver = EnhancedRLSolver(uavs, targets, graph, obstacles, self.config, 
                                    network_type=network_type, use_ph_rrt=True)
            
            # 训练
            start_time = time.time()
            history = solver.train(episodes, patience=30, log_interval=20)
            training_time = time.time() - start_time
            
            # 获取任务分配结果
            assignments = solver.get_task_assignments()
            
            # 评估解质量（简化评估）
            evaluation_metrics = self._evaluate_solution_quality(assignments, uavs, targets)
            
            # 保存结果
            quality_results[network_type] = {
                'history': history,
                'training_time': training_time,
                'assignments': assignments,
                'evaluation_metrics': evaluation_metrics,
                'final_avg_reward': np.mean(history['episode_rewards'][-50:]),
                'max_reward': max(history['episode_rewards']),
                'convergence_episode': len(history['episode_rewards']),
                'avg_deadlock_events': np.mean(history['deadlock_events']),
                'avg_resource_satisfaction': np.mean(history['resource_satisfaction'])
            }
            
            print(f"  训练时间: {training_time:.2f}秒")
            print(f"  最终平均奖励: {quality_results[network_type]['final_avg_reward']:.2f}")
            print(f"  最大奖励: {quality_results[network_type]['max_reward']:.2f}")
            print(f"  收敛轮次: {quality_results[network_type]['convergence_episode']}")
            print(f"  平均死锁事件: {quality_results[network_type]['avg_deadlock_events']:.2f}")
            print(f"  平均资源满足率: {quality_results[network_type]['avg_resource_satisfaction']:.3f}")
            print(f"  解质量评估:")
            for metric, value in evaluation_metrics.items():
                print(f"    {metric}: {value:.3f}")
        
        self.results['quality'] = quality_results
        return quality_results
    
    def _evaluate_solution_quality(self, assignments, uavs, targets):
        """简化解质量评估"""
        # 计算基本指标
        total_assignments = len(assignments)
        completed_targets = len(set(assignment['target_id'] for assignment in assignments))
        total_targets = len(targets)
        
        # 计算资源效率
        total_resource_contribution = sum(assignment.get('actual_contribution', 0) for assignment in assignments)
        total_initial_resources = sum(np.sum(uav.initial_resources) for uav in uavs)
        resource_efficiency = total_resource_contribution / total_initial_resources if total_initial_resources > 0 else 0
        
        # 计算路径效率
        total_path_length = sum(assignment.get('path_length', 0) for assignment in assignments)
        avg_path_length = total_path_length / len(assignments) if assignments else 0
        
        # 计算协作分数
        target_assignments = {}
        for assignment in assignments:
            target_id = assignment['target_id']
            if target_id not in target_assignments:
                target_assignments[target_id] = 0
            target_assignments[target_id] += 1
        
        collaboration_score = sum(1 for count in target_assignments.values() if count > 1) / len(target_assignments) if target_assignments else 0
        
        # 计算死锁避免分数
        deadlock_avoidance = 1.0 - (len([a for a in assignments if a.get('invalid_action', False)]) / len(assignments) if assignments else 0)
        
        # 综合评分
        overall_score = (
            (completed_targets / total_targets) * 0.3 +
            resource_efficiency * 0.25 +
            (1.0 / (avg_path_length + 1e-6)) * 0.2 +
            collaboration_score * 0.15 +
            deadlock_avoidance * 0.1
        )
        
        return {
            'resource_efficiency': resource_efficiency,
            'path_efficiency': 1.0 / (avg_path_length + 1e-6),
            'collaboration_score': collaboration_score,
            'deadlock_avoidance': deadlock_avoidance,
            'overall_score': overall_score
        }
    
    def plot_convergence_comparison(self, save_path=None):
        """绘制收敛性对比图"""
        if 'convergence' not in self.results:
            print("没有收敛性测试结果")
            return
        
        convergence_results = self.results['convergence']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 奖励曲线对比
        for network_type in self.network_types:
            history = convergence_results[network_type]['history']
            axes[0, 0].plot(history['episode_rewards'], label=network_type, alpha=0.7)
        axes[0, 0].set_title('训练奖励曲线对比')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 探索率对比
        for network_type in self.network_types:
            history = convergence_results[network_type]['history']
            axes[0, 1].plot(history['epsilon_history'], label=network_type, alpha=0.7)
        axes[0, 1].set_title('探索率衰减对比')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Epsilon')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 死锁事件对比
        for network_type in self.network_types:
            history = convergence_results[network_type]['history']
            axes[0, 2].plot(history['deadlock_events'], label=network_type, alpha=0.7)
        axes[0, 2].set_title('死锁事件对比')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Deadlock Count')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 资源满足率对比
        for network_type in self.network_types:
            history = convergence_results[network_type]['history']
            axes[1, 0].plot(history['resource_satisfaction'], label=network_type, alpha=0.7)
        axes[1, 0].set_title('资源满足率对比')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Satisfaction Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 平均奖励对比（滑动窗口）
        window_size = 20
        for network_type in self.network_types:
            history = convergence_results[network_type]['history']
            if len(history['episode_rewards']) >= window_size:
                avg_rewards = []
                for i in range(window_size, len(history['episode_rewards'])):
                    avg_rewards.append(np.mean(history['episode_rewards'][i-window_size:i]))
                axes[1, 1].plot(avg_rewards, label=network_type, alpha=0.7)
        axes[1, 1].set_title(f'平均奖励对比 (窗口={window_size})')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Average Reward')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 性能指标对比
        metrics = ['Final Avg Reward', 'Max Reward', 'Deadlock Events', 'Resource Satisfaction']
        network_values = {}
        for network_type in self.network_types:
            network_values[network_type] = [
                convergence_results[network_type]['final_avg_reward'],
                convergence_results[network_type]['max_reward'],
                convergence_results[network_type]['avg_deadlock_events'],
                convergence_results[network_type]['avg_resource_satisfaction']
            ]
        
        x = np.arange(len(metrics))
        width = 0.25
        
        for i, network_type in enumerate(self.network_types):
            axes[1, 2].bar(x + i*width, network_values[network_type], width, 
                          label=network_type, alpha=0.7)
        
        axes[1, 2].set_title('性能指标对比')
        axes[1, 2].set_xticks(x + width)
        axes[1, 2].set_xticklabels(metrics, rotation=45)
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_quality_comparison(self, save_path=None):
        """绘制解质量对比图"""
        if 'quality' not in self.results:
            print("没有解质量测试结果")
            return
        
        quality_results = self.results['quality']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 训练奖励曲线对比
        for network_type in self.network_types:
            history = quality_results[network_type]['history']
            axes[0, 0].plot(history['episode_rewards'], label=network_type, alpha=0.7)
        axes[0, 0].set_title('战略场景训练奖励曲线')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 解质量指标对比
        evaluation_metrics = ['resource_efficiency', 'path_efficiency', 'collaboration_score', 
                            'deadlock_avoidance', 'overall_score']
        
        for i, metric in enumerate(evaluation_metrics):
            values = []
            for network_type in self.network_types:
                metrics = quality_results[network_type]['evaluation_metrics']
                values.append(metrics.get(metric, 0.0))
            
            axes[0, 1].bar([f'{network_type}\n{metric}' for network_type in self.network_types], 
                           values, alpha=0.7)
        
        axes[0, 1].set_title('解质量指标对比')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 训练时间对比
        training_times = [quality_results[network_type]['training_time'] 
                         for network_type in self.network_types]
        axes[0, 2].bar(self.network_types, training_times, alpha=0.7)
        axes[0, 2].set_title('训练时间对比')
        axes[0, 2].set_ylabel('Time (seconds)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 收敛性能对比
        convergence_episodes = [quality_results[network_type]['convergence_episode'] 
                              for network_type in self.network_types]
        axes[1, 0].bar(self.network_types, convergence_episodes, alpha=0.7)
        axes[1, 0].set_title('收敛轮次对比')
        axes[1, 0].set_ylabel('Episodes')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 死锁控制对比
        deadlock_events = [quality_results[network_type]['avg_deadlock_events'] 
                          for network_type in self.network_types]
        axes[1, 1].bar(self.network_types, deadlock_events, alpha=0.7)
        axes[1, 1].set_title('平均死锁事件对比')
        axes[1, 1].set_ylabel('Deadlock Count')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 资源满足率对比
        resource_satisfaction = [quality_results[network_type]['avg_resource_satisfaction'] 
                               for network_type in self.network_types]
        axes[1, 2].bar(self.network_types, resource_satisfaction, alpha=0.7)
        axes[1, 2].set_title('平均资源满足率对比')
        axes[1, 2].set_ylabel('Satisfaction Rate')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_comprehensive_report(self):
        """生成综合报告"""
        print("\n=== 网络结构对比综合报告 ===")
        
        # 收敛性测试结果
        if 'convergence' in self.results:
            print("\n1. 简单场景训练收敛性测试结果:")
            convergence_results = self.results['convergence']
            
            for network_type in self.network_types:
                result = convergence_results[network_type]
                print(f"\n{network_type}:")
                print(f"  最终平均奖励: {result['final_avg_reward']:.2f}")
                print(f"  最大奖励: {result['max_reward']:.2f}")
                print(f"  收敛轮次: {result['convergence_episode']}")
                print(f"  训练时间: {result['training_time']:.2f}秒")
                print(f"  平均死锁事件: {result['avg_deadlock_events']:.2f}")
                print(f"  平均资源满足率: {result['avg_resource_satisfaction']:.3f}")
        
        # 解质量测试结果
        if 'quality' in self.results:
            print("\n2. 战略陷阱场景解质量测试结果:")
            quality_results = self.results['quality']
            
            for network_type in self.network_types:
                result = quality_results[network_type]
                print(f"\n{network_type}:")
                print(f"  最终平均奖励: {result['final_avg_reward']:.2f}")
                print(f"  最大奖励: {result['max_reward']:.2f}")
                print(f"  收敛轮次: {result['convergence_episode']}")
                print(f"  训练时间: {result['training_time']:.2f}秒")
                print(f"  平均死锁事件: {result['avg_deadlock_events']:.2f}")
                print(f"  平均资源满足率: {result['avg_resource_satisfaction']:.3f}")
                print(f"  解质量评估:")
                for metric, value in result['evaluation_metrics'].items():
                    print(f"    {metric}: {value:.3f}")
        
        # 综合排名
        print("\n3. 综合性能排名:")
        self._print_ranking()
    
    def _print_ranking(self):
        """打印综合排名"""
        rankings = {}
        
        # 收敛性排名
        if 'convergence' in self.results:
            convergence_results = self.results['convergence']
            
            # 按最终平均奖励排名
            reward_ranking = sorted(self.network_types, 
                                  key=lambda x: convergence_results[x]['final_avg_reward'], 
                                  reverse=True)
            print(f"  收敛性排名 (按最终平均奖励):")
            for i, network_type in enumerate(reward_ranking, 1):
                reward = convergence_results[network_type]['final_avg_reward']
                print(f"    {i}. {network_type}: {reward:.2f}")
            
            # 按收敛速度排名
            speed_ranking = sorted(self.network_types, 
                                 key=lambda x: convergence_results[x]['convergence_episode'])
            print(f"  收敛速度排名 (按收敛轮次):")
            for i, network_type in enumerate(speed_ranking, 1):
                episodes = convergence_results[network_type]['convergence_episode']
                print(f"    {i}. {network_type}: {episodes}轮")
        
        # 解质量排名
        if 'quality' in self.results:
            quality_results = self.results['quality']
            
            # 按整体评分排名
            overall_ranking = sorted(self.network_types, 
                                   key=lambda x: quality_results[x]['evaluation_metrics'].get('overall_score', 0), 
                                   reverse=True)
            print(f"  解质量排名 (按整体评分):")
            for i, network_type in enumerate(overall_ranking, 1):
                score = quality_results[network_type]['evaluation_metrics'].get('overall_score', 0)
                print(f"    {i}. {network_type}: {score:.3f}")
    
    def save_results(self, filename='network_comparison_results.pkl'):
        """保存测试结果"""
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"测试结果已保存到: {filename}")
    
    def run_complete_test(self):
        """运行完整的网络结构对比测试"""
        print("开始网络结构对比测试...")
        
        # 1. 简单场景收敛性测试
        self.test_convergence_simple_scenario(episodes=300)
        
        # 2. 战略陷阱场景解质量测试
        self.test_solution_quality_strategic_scenario(episodes=200)
        
        # 3. 绘制对比图
        self.plot_convergence_comparison('output/images/network_convergence_comparison.png')
        self.plot_quality_comparison('output/images/network_quality_comparison.png')
        
        # 4. 生成综合报告
        self.generate_comprehensive_report()
        
        # 5. 保存结果
        self.save_results()
        
        print("\n网络结构对比测试完成！")

if __name__ == "__main__":
    # 创建测试器并运行完整测试
    tester = NetworkStructureTester()
    tester.run_complete_test() 