#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
战略场景测试脚本
使用优化后的网络在战略场景下进行试验
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import *
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

def create_strategic_scenario():
    """创建战略场景"""
    print("=== 创建战略场景 ===")
    
    # 创建多个无人机
    uavs = []
    for i in range(4):  # 4个无人机
        uav = UAV(
            id=i+1,
            position=np.array([100 + i*50, 100 + i*50, 50]),
            heading=0.0,  # 初始朝向
            resources=np.array([100, 80, 60]),  # 多种资源
            max_distance=500,
            velocity_range=(20, 30),
            economic_speed=25.0  # 经济速度
        )
        uavs.append(uav)
    
    # 创建多个目标
    targets = []
    for i in range(6):  # 6个目标
        target = Target(
            id=i+1,
            position=np.array([200 + i*30, 200 + i*30, 0]),
            resources=np.array([50, 40, 30]),  # 多种资源需求
            value=1.0 + i*0.1  # 不同价值
        )
        targets.append(target)
    
    # 创建障碍物
    from path_planning import CircularObstacle
    obstacles = [
        CircularObstacle(np.array([150, 150, 0]), 30, 5),
        CircularObstacle(np.array([250, 250, 0]), 25, 5),
        CircularObstacle(np.array([350, 150, 0]), 35, 5)
    ]
    
    print(f"战略场景创建完成:")
    print(f"  - 无人机数量: {len(uavs)}")
    print(f"  - 目标数量: {len(targets)}")
    print(f"  - 障碍物数量: {len(obstacles)}")
    print(f"  - 场景特点: 复杂多目标协同任务分配")
    
    return uavs, targets, obstacles

def calculate_state_dimension(uavs, targets, graph):
    """计算状态向量的实际维度"""
    state_dim = 0
    
    # 目标信息：位置(3) + 剩余资源(3) + 总资源(3) = 9
    for t in targets:
        state_dim += 9
    
    # 无人机信息：位置(3) + 资源(3) + 航向(1) + 距离(1) = 8
    for u in uavs:
        state_dim += 8
    
    # 协同信息：每个目标4个指标
    for t in targets:
        state_dim += 4  # 拥挤度、完成度、边际效用、紧迫度
    
    # 全局协同信息：总体完成度(1) + 均衡度(1) = 2
    state_dim += 2
    
    return state_dim

def calculate_output_dimension(uavs, targets, graph):
    """计算输出维度的实际大小"""
    return len(targets) * len(uavs) * len(graph.phi_set)

def test_optimized_networks_in_strategic_scenario():
    """在战略场景下测试优化网络"""
    print("\n=== 战略场景优化网络测试 ===")
    
    # 创建配置
    config = Config()
    
    # 创建战略场景
    uavs, targets, obstacles = create_strategic_scenario()
    graph = DirectedGraph(uavs, targets, config.GRAPH_N_PHI, obstacles)
    
    # 计算正确的维度
    input_dim = calculate_state_dimension(uavs, targets, graph)
    output_dim = calculate_output_dimension(uavs, targets, graph)
    
    print(f"计算得到的维度:")
    print(f"  - 输入维度: {input_dim}")
    print(f"  - 输出维度: {output_dim}")
    print(f"  - 隐藏层: [128, 64, 32]")
    
    # 测试不同网络类型
    networks = [
        ('OptimizedDeepFCN', '优化DeepFCN'),
        ('OptimizedDeepFCN_Residual', '优化残差网络'),
        ('OptimizedGNN', '优化GNN')
    ]
    
    results = {}
    
    for network_type, network_name in networks:
        print(f"\n{'='*60}")
        print(f"测试 {network_name} 在战略场景下的表现")
        print(f"{'='*60}")
        
        try:
            # 创建求解器 - 使用正确的维度
            solver = GraphRLSolver(uavs, targets, graph, obstacles, 
                                  input_dim, [128, 64, 32], output_dim, config, network_type)
            
            print(f"✅ {network_name} 创建成功")
            
            # 收集战略场景经验
            print("收集战略场景训练经验...")
            episode_count = 0
            while len(solver.memory) < solver.config.BATCH_SIZE and episode_count < 100:
                state = solver.env.reset()
                step_count = 0
                episode_reward = 0
                
                while step_count < 50:  # 增加步数以适应复杂场景
                    valid_actions = solver._get_valid_action_mask()
                    if not valid_actions:
                        break
                        
                    # 使用epsilon-greedy策略 - 修复索引问题
                    if np.random.random() < solver.epsilon:
                        # 修复：直接选择随机动作，而不是用索引
                        action = valid_actions[np.random.randint(len(valid_actions))]
                    else:
                        # 选择Q值最大的动作
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(solver.device)
                        with torch.no_grad():
                            q_values = solver.q_network(state_tensor)
                            # 只考虑有效动作的Q值
                            valid_indices = [solver._action_to_index(a) for a in valid_actions]
                            valid_q_values = q_values[0, valid_indices]
                            max_idx = torch.argmax(valid_q_values).item()
                            action = valid_actions[max_idx]
                    
                    next_state, reward, done, info = solver.env.step(action)
                    episode_reward += reward
                    
                    # 存储经验
                    solver.memory.append((state, action, reward, next_state, done))
                    state = next_state
                    step_count += 1
                    
                    if done:
                        break
                
                episode_count += 1
                if episode_count % 20 == 0:
                    print(f"已收集 {len(solver.memory)} 个经验，平均奖励: {episode_reward/step_count:.2f}")
            
            print(f"✅ 经验收集完成: {len(solver.memory)} 个经验")
            
            # 测试训练稳定性
            print("测试训练稳定性...")
            grad_norms = []
            losses = []
            
            for i in range(20):  # 增加训练次数
                try:
                    loss = solver.replay()
                    if loss is not None:
                        losses.append(loss)
                        
                        # 计算梯度范数
                        grad_norm = solver._calculate_gradient_norm()
                        grad_norms.append(grad_norm)
                        
                        if grad_norm > 20:  # 更宽松的阈值
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
                
                print(f"\n{network_name} 战略场景训练结果:")
                print(f"  平均梯度范数: {avg_grad_norm:.4f}")
                print(f"  最大梯度范数: {max_grad_norm:.4f}")
                print(f"  最小梯度范数: {min_grad_norm:.4f}")
                print(f"  平均损失: {avg_loss:.4f}")
                
                # 判断稳定性
                is_stable = True
                if max_grad_norm > 100:  # 更宽松的阈值
                    print(f"  ⚠️  梯度爆炸风险")
                    is_stable = False
                if min_grad_norm < 1e-8:
                    print(f"  ⚠️  梯度消失风险")
                    is_stable = False
                if avg_grad_norm > 50:  # 更宽松的阈值
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
    
    return results

def run_strategic_training():
    """运行战略场景完整训练"""
    print("\n=== 战略场景完整训练 ===")
    
    # 创建配置
    config = Config()
    config.EPISODES = 500  # 增加训练轮次
    config.LEARNING_RATE = 0.0005  # 使用优化学习率
    config.BATCH_SIZE = 64  # 减小批次大小
    config.MEMORY_SIZE = 10000  # 增加记忆库大小
    
    # 创建战略场景
    uavs, targets, obstacles = create_strategic_scenario()
    graph = DirectedGraph(uavs, targets, config.GRAPH_N_PHI, obstacles)
    
    # 计算正确的维度
    input_dim = calculate_state_dimension(uavs, targets, graph)
    output_dim = calculate_output_dimension(uavs, targets, graph)
    
    # 选择最佳网络进行完整训练
    best_network = 'OptimizedDeepFCN'  # 根据之前测试结果选择
    
    print(f"使用 {best_network} 进行战略场景完整训练")
    print(f"输入维度: {input_dim}, 输出维度: {output_dim}")
    
    try:
        # 创建求解器 - 使用正确的维度
        solver = GraphRLSolver(uavs, targets, graph, obstacles, 
                              input_dim, [128, 64, 32], output_dim, config, best_network)
        
        # 开始训练
        start_time = time.time()
        training_time = solver.train(
            episodes=config.EPISODES,
            patience=100,
            log_interval=20,
            model_save_path=f"output/models/strategic_{best_network}_model.pth"
        )
        
        print(f"✅ 战略场景训练完成，耗时: {training_time:.2f}秒")
        
        # 测试训练后的模型
        print("测试训练后的模型...")
        final_plan = solver.get_task_assignments()
        
        # 评估结果
        from evaluate import evaluate_plan
        evaluation_metrics = evaluate_plan(final_plan, uavs, targets)
        
        print(f"✅ 战略场景训练和评估完成")
        print(f"评估指标: {evaluation_metrics}")
        
    except Exception as e:
        print(f"❌ 战略场景训练失败: {e}")
        import traceback
        traceback.print_exc()

def plot_strategic_results(results):
    """绘制战略场景测试结果"""
    if not results:
        print("没有结果可绘制")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('战略场景优化网络测试结果', fontsize=16)
    
    # 提取数据
    network_names = list(results.keys())
    avg_grad_norms = [results[name]['avg_grad_norm'] for name in network_names]
    max_grad_norms = [results[name]['max_grad_norm'] for name in network_names]
    avg_losses = [results[name]['avg_loss'] for name in network_names]
    stability_scores = [1 if results[name]['is_stable'] else 0 for name in network_names]
    
    # 平均梯度范数对比
    axes[0, 0].bar(network_names, avg_grad_norms, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[0, 0].set_title('平均梯度范数对比')
    axes[0, 0].set_ylabel('梯度范数')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 最大梯度范数对比
    axes[0, 1].bar(network_names, max_grad_norms, color=['#FF8E8E', '#6EDDD6', '#6BC5E3'])
    axes[0, 1].set_title('最大梯度范数对比')
    axes[0, 1].set_ylabel('梯度范数')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 平均损失对比
    axes[1, 0].bar(network_names, avg_losses, color=['#FFB3B3', '#9EE8E1', '#91D3E9'])
    axes[1, 0].set_title('平均损失对比')
    axes[1, 0].set_ylabel('损失值')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 稳定性评分
    axes[1, 1].bar(network_names, stability_scores, color=['#4CAF50', '#FF9800', '#F44336'])
    axes[1, 1].set_title('训练稳定性评分')
    axes[1, 1].set_ylabel('稳定性 (1=稳定, 0=不稳定)')
    axes[1, 1].set_ylim(0, 1.2)
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for i, (ax, values) in enumerate([(axes[0, 0], avg_grad_norms), 
                                     (axes[0, 1], max_grad_norms),
                                     (axes[1, 0], avg_losses)]):
        for j, v in enumerate(values):
            ax.text(j, v + max(values)*0.01, f'{v:.3f}', ha='center', va='bottom')
    
    for i, v in enumerate(stability_scores):
        status = "稳定" if v == 1 else "不稳定"
        axes[1, 1].text(i, v + 0.1, status, ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('output/strategic_scenario_test_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ 战略场景测试结果图表已保存")

if __name__ == "__main__":
    print("=== 战略场景优化网络测试 ===")
    
    # 测试优化网络在战略场景下的表现
    results = test_optimized_networks_in_strategic_scenario()
    
    # 绘制结果
    plot_strategic_results(results)
    
    # 运行完整训练
    run_strategic_training()
    
    print("✅ 战略场景测试完成") 