#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试优化网络训练效果
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import *
import torch
import numpy as np
import matplotlib.pyplot as plt

def test_optimized_networks():
    """测试优化网络的效果"""
    print("=== 测试优化网络训练效果 ===")
    
    # 创建配置
    config = Config()
    
    # 创建简单场景
    uavs, targets, obstacles = get_simple_scenario(obstacle_tolerance=0.1)
    graph = DirectedGraph(uavs, targets, config.GRAPH_N_PHI, obstacles)
    
    # 测试不同网络类型
    networks = [
        ('DeepFCN', '原始DeepFCN'),
        ('OptimizedDeepFCN', '优化DeepFCN'),
        ('DeepFCN_Residual', '原始残差网络'),
        ('OptimizedDeepFCN_Residual', '优化残差网络'),
        ('GNN', '原始GNN'),
        ('OptimizedGNN', '优化GNN')
    ]
    
    results = {}
    
    for network_type, network_name in networks:
        print(f"\n{'='*50}")
        print(f"测试 {network_name}")
        print(f"{'='*50}")
        
        try:
            # 创建求解器
            solver = GraphRLSolver(uavs, targets, graph, obstacles, 
                                  34, [128, 64, 32], 24, config, network_type)
            
            print(f"✅ {network_name} 创建成功")
            
            # 收集经验
            print("收集训练经验...")
            episode_count = 0
            while len(solver.memory) < solver.config.BATCH_SIZE and episode_count < 50:
                state = solver.env.reset()
                step_count = 0
                
                while step_count < 20:  # 限制每轮步数
                    valid_actions = solver._get_valid_action_mask()
                    if not valid_actions:
                        break
                        
                    action = valid_actions[0]  # 简单选择第一个有效动作
                    next_state, reward, done, info = solver.env.step(action)
                    
                    # 存储经验
                    solver.memory.append((state, action, reward, next_state, done))
                    state = next_state
                    step_count += 1
                    
                    if done:
                        break
                
                episode_count += 1
                if episode_count % 10 == 0:
                    print(f"已收集 {len(solver.memory)} 个经验")
            
            print(f"✅ 经验收集完成: {len(solver.memory)} 个经验")
            
            # 测试训练稳定性
            print("测试训练稳定性...")
            grad_norms = []
            losses = []
            
            for i in range(10):  # 进行10次训练
                try:
                    loss = solver.replay()
                    if loss is not None:
                        losses.append(loss)
                        
                        # 计算梯度范数
                        grad_norm = solver._calculate_gradient_norm()
                        grad_norms.append(grad_norm)
                        
                        if grad_norm > 10:
                            print(f"⚠️  第{i+1}次训练: 梯度范数过大 ({grad_norm:.4f})")
                        elif grad_norm < 1e-6:
                            print(f"⚠️  第{i+1}次训练: 梯度范数过小 ({grad_norm:.6f})")
                        else:
                            print(f"✅ 第{i+1}次训练: 梯度范数正常 ({grad_norm:.4f})")
                    
                except Exception as e:
                    print(f"❌ 第{i+1}次训练失败: {e}")
                    break
            
            # 分析结果
            if grad_norms and losses:
                avg_grad_norm = np.mean(grad_norms)
                avg_loss = np.mean(losses)
                max_grad_norm = np.max(grad_norms)
                min_grad_norm = np.min(grad_norms)
                
                print(f"\n{network_name} 训练结果:")
                print(f"  平均梯度范数: {avg_grad_norm:.4f}")
                print(f"  最大梯度范数: {max_grad_norm:.4f}")
                print(f"  最小梯度范数: {min_grad_norm:.4f}")
                print(f"  平均损失: {avg_loss:.4f}")
                
                # 判断稳定性
                is_stable = True
                if max_grad_norm > 50:
                    print(f"  ⚠️  梯度爆炸风险")
                    is_stable = False
                if min_grad_norm < 1e-8:
                    print(f"  ⚠️  梯度消失风险")
                    is_stable = False
                if avg_grad_norm > 20:
                    print(f"  ⚠️  平均梯度范数过大")
                    is_stable = False
                
                if is_stable:
                    print(f"  ✅ 训练稳定")
                else:
                    print(f"  ❌ 训练不稳定")
                
                results[network_name] = {
                    'avg_grad_norm': avg_grad_norm,
                    'max_grad_norm': max_grad_norm,
                    'min_grad_norm': min_grad_norm,
                    'avg_loss': avg_loss,
                    'is_stable': is_stable
                }
            
        except Exception as e:
            print(f"❌ {network_name} 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 生成对比报告
    print("\n" + "="*60)
    print("优化网络效果对比报告")
    print("="*60)
    
    if results:
        # 按稳定性排序
        stable_networks = [(name, data) for name, data in results.items() if data['is_stable']]
        unstable_networks = [(name, data) for name, data in results.items() if not data['is_stable']]
        
        print(f"\n稳定网络 ({len(stable_networks)}个):")
        for name, data in stable_networks:
            print(f"  {name}: 平均梯度范数={data['avg_grad_norm']:.4f}, 平均损失={data['avg_loss']:.4f}")
        
        if unstable_networks:
            print(f"\n不稳定网络 ({len(unstable_networks)}个):")
            for name, data in unstable_networks:
                print(f"  {name}: 平均梯度范数={data['avg_grad_norm']:.4f}, 平均损失={data['avg_loss']:.4f}")
        
        # 找出最佳网络
        if stable_networks:
            best_network = min(stable_networks, key=lambda x: x[1]['avg_grad_norm'])
            print(f"\n🎉 最佳稳定网络: {best_network[0]}")
            print(f"   平均梯度范数: {best_network[1]['avg_grad_norm']:.4f}")
            print(f"   平均损失: {best_network[1]['avg_loss']:.4f}")
    
    return results

def plot_comparison_results(results):
    """绘制对比结果"""
    if not results:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('优化网络效果对比', fontsize=16, fontweight='bold')
    
    # 准备数据
    network_names = list(results.keys())
    avg_grad_norms = [results[name]['avg_grad_norm'] for name in network_names]
    max_grad_norms = [results[name]['max_grad_norm'] for name in network_names]
    avg_losses = [results[name]['avg_loss'] for name in network_names]
    is_stable = [results[name]['is_stable'] for name in network_names]
    
    # 1. 平均梯度范数对比
    ax1 = axes[0, 0]
    colors = ['green' if stable else 'red' for stable in is_stable]
    bars1 = ax1.bar(network_names, avg_grad_norms, color=colors, alpha=0.7)
    ax1.set_title('平均梯度范数对比')
    ax1.set_ylabel('梯度范数')
    ax1.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, value in zip(bars1, avg_grad_norms):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{value:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 2. 最大梯度范数对比
    ax2 = axes[0, 1]
    bars2 = ax2.bar(network_names, max_grad_norms, color=colors, alpha=0.7)
    ax2.set_title('最大梯度范数对比')
    ax2.set_ylabel('梯度范数')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars2, max_grad_norms):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 3. 平均损失对比
    ax3 = axes[1, 0]
    bars3 = ax3.bar(network_names, avg_losses, color=colors, alpha=0.7)
    ax3.set_title('平均损失对比')
    ax3.set_ylabel('损失值')
    ax3.tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars3, avg_losses):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{value:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 4. 稳定性统计
    ax4 = axes[1, 1]
    stable_count = sum(is_stable)
    unstable_count = len(is_stable) - stable_count
    ax4.pie([stable_count, unstable_count], labels=['稳定', '不稳定'], 
            autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
    ax4.set_title('网络稳定性统计')
    
    plt.tight_layout()
    plt.savefig('output/optimized_networks_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 对比图表已保存至: output/optimized_networks_comparison.png")

if __name__ == "__main__":
    # 创建输出目录
    os.makedirs('output', exist_ok=True)
    
    # 测试优化网络
    results = test_optimized_networks()
    
    # 绘制对比结果
    if results:
        plot_comparison_results(results)
    
    print("\n🎉 优化网络测试完成!") 