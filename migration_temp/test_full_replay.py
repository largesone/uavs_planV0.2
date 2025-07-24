#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整测试replay功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import *
import torch
import numpy as np

def test_full_replay():
    """完整测试replay功能"""
    print("=== 完整测试replay功能 ===")
    
    # 创建简单场景
    config = Config()
    uavs, targets, obstacles = get_simple_scenario(obstacle_tolerance=0.1)
    graph = DirectedGraph(uavs, targets, config.GRAPH_N_PHI, obstacles)
    
    # 创建求解器
    solver = GraphRLSolver(uavs, targets, graph, obstacles, 
                          34, [256, 128, 64], 24, config, 'DeepFCN')
    
    print("✅ 求解器创建成功")
    
    # 收集足够的经验
    print("收集经验...")
    episode_count = 0
    while len(solver.memory) < solver.config.BATCH_SIZE and episode_count < 100:
        state = solver.env.reset()
        step_count = 0
        
        while step_count < 50:  # 限制每轮步数
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
    
    # 测试replay
    try:
        loss = solver.replay()
        print(f"✅ replay成功，损失: {loss}")
    except Exception as e:
        print(f"❌ replay失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试不同网络类型
    networks = ['DeepFCN', 'DeepFCN_Residual', 'GNN']
    
    for network_type in networks:
        print(f"\n=== 测试 {network_type} replay ===")
        try:
            solver_test = GraphRLSolver(uavs, targets, graph, obstacles, 
                                       34, [256, 128, 64], 24, config, network_type)
            
            # 复制经验到测试求解器
            solver_test.memory = solver.memory.copy()
            print(f"✅ {network_type} 创建成功，复制了 {len(solver_test.memory)} 个经验")
            
            # 测试replay
            loss = solver_test.replay()
            print(f"✅ {network_type} replay成功，损失: {loss}")
            
        except Exception as e:
            print(f"❌ {network_type} 测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n✅ 所有网络replay测试通过!")
    return True

if __name__ == "__main__":
    success = test_full_replay()
    if success:
        print("\n🎉 所有测试通过，可以运行批量测试了!")
    else:
        print("\n❌ 测试失败，需要进一步修复") 