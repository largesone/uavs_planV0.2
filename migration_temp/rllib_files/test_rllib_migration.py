# -*- coding: utf-8 -*-
# 文件名: test_rllib_migration.py
# 描述: 测试RLlib迁移的基本功能

import os
import sys
import numpy as np
import time

# 导入项目模块
from entities import UAV, Target
from scenarios import get_simple_scenario
from config import Config
from rllib_env import UAVTaskEnvRLlib

def test_environment_creation():
    """测试环境创建"""
    print("=== 测试环境创建 ===")
    
    # 创建简单场景
    uavs, targets, obstacles = get_simple_scenario(50.0)
    
    # 创建有向图
    from main import DirectedGraph
    graph = DirectedGraph(uavs, targets, 6, obstacles)
    
    # 创建配置
    config = Config()
    
    # 创建RLlib环境
    env = UAVTaskEnvRLlib(uavs, targets, graph, obstacles, config)
    
    print(f"✓ 环境创建成功")
    print(f"  观察空间: {env.observation_space}")
    print(f"  动作空间: {env.action_space}")
    print(f"  状态维度: {env.state_dim}")
    print(f"  动作维度: {env.action_dim}")
    
    return env

def test_environment_reset():
    """测试环境重置"""
    print("\n=== 测试环境重置 ===")
    
    env = test_environment_creation()
    
    # 重置环境
    obs, info = env.reset()
    
    print(f"✓ 环境重置成功")
    print(f"  观察形状: {obs.shape}")
    print(f"  观察类型: {type(obs)}")
    print(f"  信息: {info}")
    
    return env, obs

def test_environment_step():
    """测试环境步进"""
    print("\n=== 测试环境步进 ===")
    
    env, obs = test_environment_reset()
    
    # 测试几个随机动作
    for i in range(5):
        # 随机选择动作
        action = np.random.randint(0, env.action_space.n)
        
        # 执行动作
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"  步骤 {i+1}:")
        print(f"    动作: {action}")
        print(f"    奖励: {reward:.2f}")
        print(f"    终止: {terminated}")
        print(f"    截断: {truncated}")
        print(f"    信息: {info}")
        
        if terminated:
            print("    环境已终止")
            break
        
        obs = next_obs
    
    return env

def test_task_assignments():
    """测试任务分配"""
    print("\n=== 测试任务分配 ===")
    
    env = test_environment_step()
    
    # 获取任务分配
    assignments = env.get_task_assignments()
    
    print(f"✓ 任务分配获取成功")
    print(f"  分配结果: {assignments}")
    
    # 统计分配情况
    total_assignments = sum(len(assignments[uav_id]) for uav_id in assignments)
    print(f"  总分配数: {total_assignments}")
    
    return assignments

def test_reward_calculation():
    """测试奖励计算"""
    print("\n=== 测试奖励计算 ===")
    
    # 创建环境
    uavs, targets, obstacles = get_simple_scenario(50.0)
    from main import DirectedGraph
    graph = DirectedGraph(uavs, targets, 6, obstacles)
    config = Config()
    env = UAVTaskEnvRLlib(uavs, targets, graph, obstacles, config)
    
    # 重置环境
    obs, info = env.reset()
    
    # 执行一些动作并观察奖励
    rewards = []
    for i in range(10):
        action = np.random.randint(0, env.action_space.n)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        
        if terminated:
            break
    
    print(f"✓ 奖励计算测试完成")
    print(f"  奖励列表: {[f'{r:.2f}' for r in rewards]}")
    print(f"  平均奖励: {np.mean(rewards):.2f}")
    print(f"  奖励范围: [{min(rewards):.2f}, {max(rewards):.2f}]")

def test_action_mapping():
    """测试动作映射"""
    print("\n=== 测试动作映射 ===")
    
    env = test_environment_creation()
    
    # 测试动作映射
    print(f"  动作空间大小: {env.action_space.n}")
    print(f"  动作映射示例:")
    
    for i in range(min(5, env.action_space.n)):
        action_tuple = env.action_to_tuple[i]
        action_idx = env.tuple_to_action[action_tuple]
        print(f"    索引 {i} -> 元组 {action_tuple} -> 索引 {action_idx}")
    
    print(f"✓ 动作映射测试完成")

def test_state_representation():
    """测试状态表示"""
    print("\n=== 测试状态表示 ===")
    
    env = test_environment_creation()
    obs, info = env.reset()
    
    print(f"✓ 状态表示测试完成")
    print(f"  状态维度: {obs.shape}")
    print(f"  状态类型: {obs.dtype}")
    print(f"  状态范围: [{obs.min():.2f}, {obs.max():.2f}]")
    print(f"  状态均值: {obs.mean():.2f}")
    print(f"  状态标准差: {obs.std():.2f}")

def run_comprehensive_test():
    """运行综合测试"""
    print("=== RLlib迁移综合测试 ===")
    print("开始测试各个组件...")
    
    try:
        # 测试环境创建
        test_environment_creation()
        
        # 测试环境重置
        test_environment_reset()
        
        # 测试环境步进
        test_environment_step()
        
        # 测试任务分配
        test_task_assignments()
        
        # 测试奖励计算
        test_reward_calculation()
        
        # 测试动作映射
        test_action_mapping()
        
        # 测试状态表示
        test_state_representation()
        
        print("\n=== 所有测试通过! ===")
        print("✓ RLlib迁移成功")
        print("✓ 环境适配正常")
        print("✓ 基本功能完整")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("RLlib迁移测试开始...")
    
    success = run_comprehensive_test()
    
    if success:
        print("\n🎉 迁移测试成功完成!")
        print("现在可以运行RLlib训练了:")
        print("python main_rllib.py --scenario simple --episodes 100")
    else:
        print("\n💥 迁移测试失败!")
        print("请检查错误信息并修复问题")

if __name__ == "__main__":
    main() 