# -*- coding: utf-8 -*-
# 文件名: gcn_rl_param_analyzer.py
# 描述: 分析GCN-RL算法参数对求解结果的影响，寻找优化的算法参数设置

import itertools
import time
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from main import Config, run_scenario
from scenarios import get_small_scenario, get_complex_scenario
from evaluate import evaluate_plan

def analyze_gcn_rl_parameters():
    """
    分析GCN-RL算法参数对求解结果的影响，重点关注EPISODES和USE_PHRRT_DURING_TRAINING参数
    """
    # 创建输出目录
    output_dir = "output/parameter_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 定义要测试的参数网格 ---
    param_grid = {
        'EPISODES': [200, 400, 800, 1200],  # 训练总轮次
        'USE_PHRRT_DURING_TRAINING': [False, True],  # 距离计算方式
        'GRAPH_N_PHI': [4, 6, 8],  # 构建图时的离散化接近角度数量
        'LEARNING_RATE': [0.0001, 0.0005, 0.001]  # 学习率
    }

    # 生成所有参数组合
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"将要测试 {len(param_combinations)} 种不同的参数组合...")

    # --- 选择测试场景 ---
    base_uavs, base_targets, base_obstacles = get_small_scenario(obstacle_tolerance=50.0)
    scenario_name_base = "ParamAnalysis_SmallScenario"

    # 准备结果存储
    results = []

    for i, params in enumerate(param_combinations):
        print(f"\n{'='*40}\n[测试 {i+1}/{len(param_combinations)}] 参数: {params}\n{'='*40}")
        
        # --- 应用参数 ---
        config = Config()
        config.RUN_TRAINING = True  # 强制进行训练
        for key, value in params.items():
            setattr(config, key, value)
        
        # 创建可读的参数字符串用于模型路径和结果标识
        param_str = f"ep{params['EPISODES']}_phrrt{params['USE_PHRRT_DURING_TRAINING']}_phi{params['GRAPH_N_PHI']}_lr{params['LEARNING_RATE']}"
        scenario_name = f"{scenario_name_base}_{param_str}"

        # --- 运行场景 ---
        start_time = time.time()
        final_plan, training_time, deadlocked_tasks = run_scenario(
            config=config,
            base_uavs=base_uavs,
            base_targets=base_targets,
            obstacles=base_obstacles,
            scenario_name=scenario_name,
            save_visualization=False,  # 批量测试时关闭可视化以加快速度
            show_visualization=False,
            save_report=False
        )
        end_time = time.time()

        # --- 评估和记录结果 ---
        total_runtime = end_time - start_time
        
        # 使用evaluate_plan函数获取详细评估指标
        evaluation_metrics = None
        if final_plan:
            evaluation_metrics = evaluate_plan(final_plan, base_uavs, base_targets, deadlocked_tasks)
            
            # 记录结果
            result = {
                'params': params,
                'training_time': training_time,
                'total_runtime': total_runtime,
                **evaluation_metrics  # 展开所有评估指标
            }
            results.append(result)
            
            print(f"--- 测试 {i+1} 完成 ---")
            print(f"训练时间: {training_time:.2f}s, 总运行时间: {total_runtime:.2f}s")
            print(f"总奖励分数: {evaluation_metrics['total_reward_score']:.2f}")
            print(f"完成率: {evaluation_metrics['completion_rate']:.4f}")
            print(f"目标满足率: {evaluation_metrics['satisfied_targets_rate']:.4f}")
            print(f"是否死锁: {'是' if evaluation_metrics['is_deadlocked'] else '否'}")
        else:
            print(f"--- 测试 {i+1} 失败 --- 未能生成有效方案")
            # 记录失败结果
            results.append({
                'params': params,
                'training_time': training_time,
                'total_runtime': total_runtime,
                'total_reward_score': -1000,
                'completion_rate': 0,
                'satisfied_targets_rate': 0,
                'is_deadlocked': 1,
                'deadlocked_uav_count': len(base_uavs),
                'resource_utilization_rate': 0,
                'sync_feasibility_rate': 0,
                'load_balance_score': 0,
                'total_distance': 0
            })

    # --- 结果分析和可视化 ---
    # 转换为DataFrame便于分析
    df = pd.DataFrame(results)
    
    # 保存原始结果到CSV
    csv_path = os.path.join(output_dir, "parameter_analysis_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n原始结果已保存至: {csv_path}")
    
    # 按总奖励分数排序
    sorted_df = df.sort_values(by='total_reward_score', ascending=False)
    
    # 打印排名前10的参数组合
    print(f"\n{'='*50}\n========= 参数测试最终排名 (前10) =========\n{'='*50}")
    for i, (_, row) in enumerate(sorted_df.head(10).iterrows()):
        print(f"\n--- 排名 {i+1} ---")
        print(f"  参数: EPISODES={row['params']['EPISODES']}, "
              f"USE_PHRRT_DURING_TRAINING={row['params']['USE_PHRRT_DURING_TRAINING']}, "
              f"GRAPH_N_PHI={row['params']['GRAPH_N_PHI']}, "
              f"LEARNING_RATE={row['params']['LEARNING_RATE']}")
        print(f"  性能: 总奖励={row['total_reward_score']:.2f}, 完成率={row['completion_rate']:.4f}, "
              f"目标满足率={row['satisfied_targets_rate']:.4f}")
        print(f"  耗时: 训练={row['training_time']:.2f}s, 总计={row['total_runtime']:.2f}s")
    
    # --- 参数影响分析 ---
    analyze_parameter_impact(df, output_dir)

def analyze_parameter_impact(df, output_dir):
    """
    分析各参数对求解结果的影响并生成可视化图表
    """
    # 1. EPISODES参数对总奖励的影响
    plt.figure(figsize=(10, 6))
    for phrrt in [False, True]:
        for phi in sorted(df['params'].apply(lambda x: x['GRAPH_N_PHI']).unique()):
            subset = df[df['params'].apply(lambda x: x['USE_PHRRT_DURING_TRAINING'] == phrrt and x['GRAPH_N_PHI'] == phi)]
            if not subset.empty:
                # 按EPISODES分组并计算平均总奖励
                grouped = subset.groupby(subset['params'].apply(lambda x: x['EPISODES']))
                episodes = []
                rewards = []
                for ep, group in grouped:
                    episodes.append(ep)
                    rewards.append(group['total_reward_score'].mean())
                
                plt.plot(episodes, rewards, marker='o', label=f"PHRRT={phrrt}, PHI={phi}")
    
    plt.xlabel('训练轮次 (EPISODES)')
    plt.ylabel('平均总奖励分数')
    plt.title('训练轮次对求解质量的影响')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "episodes_impact.png"), dpi=300)
    
    # 2. USE_PHRRT_DURING_TRAINING参数对训练时间和求解质量的影响
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    
    # 训练时间比较
    phrrt_false = df[df['params'].apply(lambda x: not x['USE_PHRRT_DURING_TRAINING'])]['training_time']
    phrrt_true = df[df['params'].apply(lambda x: x['USE_PHRRT_DURING_TRAINING'])]['training_time']
    
    labels = ['快速近似距离 (False)', '高精度PH-RRT (True)']
    plt.boxplot([phrrt_false, phrrt_true], labels=labels)
    plt.ylabel('训练时间 (秒)')
    plt.title('距离计算方式对训练时间的影响')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 求解质量比较
    plt.subplot(1, 2, 2)
    phrrt_false_reward = df[df['params'].apply(lambda x: not x['USE_PHRRT_DURING_TRAINING'])]['total_reward_score']
    phrrt_true_reward = df[df['params'].apply(lambda x: x['USE_PHRRT_DURING_TRAINING'])]['total_reward_score']
    
    plt.boxplot([phrrt_false_reward, phrrt_true_reward], labels=labels)
    plt.ylabel('总奖励分数')
    plt.title('距离计算方式对求解质量的影响')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "phrrt_impact.png"), dpi=300)
    
    # 3. 参数组合热力图 - EPISODES vs GRAPH_N_PHI (对于每种PHRRT设置)
    for phrrt in [False, True]:
        plt.figure(figsize=(10, 8))
        
        # 筛选数据
        subset = df[df['params'].apply(lambda x: x['USE_PHRRT_DURING_TRAINING'] == phrrt)]
        
        # 创建透视表
        pivot_data = {}
        for _, row in subset.iterrows():
            ep = row['params']['EPISODES']
            phi = row['params']['GRAPH_N_PHI']
            lr = row['params']['LEARNING_RATE']
            key = (ep, phi, lr)
            if key not in pivot_data:
                pivot_data[key] = []
            pivot_data[key].append(row['total_reward_score'])
        
        # 计算平均值
        for key in pivot_data:
            pivot_data[key] = np.mean(pivot_data[key])
        
        # 转换为热力图数据
        episodes = sorted(subset['params'].apply(lambda x: x['EPISODES']).unique())
        phis = sorted(subset['params'].apply(lambda x: x['GRAPH_N_PHI']).unique())
        lrs = sorted(subset['params'].apply(lambda x: x['LEARNING_RATE']).unique())
        
        # 为每个学习率创建一个子图
        fig, axes = plt.subplots(1, len(lrs), figsize=(15, 5), sharey=True)
        if len(lrs) == 1:
            axes = [axes]
            
        for i, lr in enumerate(lrs):
            heatmap_data = np.zeros((len(episodes), len(phis)))
            for e_idx, ep in enumerate(episodes):
                for p_idx, phi in enumerate(phis):
                    key = (ep, phi, lr)
                    if key in pivot_data:
                        heatmap_data[e_idx, p_idx] = pivot_data[key]
            
            im = axes[i].imshow(heatmap_data, cmap='viridis')
            axes[i].set_title(f'学习率 = {lr}')
            axes[i].set_xticks(np.arange(len(phis)))
            axes[i].set_yticks(np.arange(len(episodes)))
            axes[i].set_xticklabels(phis)
            axes[i].set_yticklabels(episodes)
            axes[i].set_xlabel('GRAPH_N_PHI')
            if i == 0:
                axes[i].set_ylabel('EPISODES')
            
            # 添加数值标签
            for e_idx, ep in enumerate(episodes):
                for p_idx, phi in enumerate(phis):
                    text = axes[i].text(p_idx, e_idx, f"{heatmap_data[e_idx, p_idx]:.0f}",
                                ha="center", va="center", color="w" if heatmap_data[e_idx, p_idx] < np.max(heatmap_data)/1.5 else "black")
        
        fig.colorbar(im, ax=axes, label='总奖励分数')
        plt.suptitle(f'参数组合热力图 (USE_PHRRT_DURING_TRAINING = {phrrt})')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"heatmap_phrrt_{phrrt}.png"), dpi=300)
    
    # 4. 训练时间与求解质量的权衡分析
    plt.figure(figsize=(10, 6))
    for phrrt in [False, True]:
        subset = df[df['params'].apply(lambda x: x['USE_PHRRT_DURING_TRAINING'] == phrrt)]
        plt.scatter(subset['training_time'], subset['total_reward_score'], 
                   alpha=0.7, label=f"PHRRT={phrrt}")
    
    plt.xlabel('训练时间 (秒)')
    plt.ylabel('总奖励分数')
    plt.title('训练时间与求解质量的权衡')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "time_quality_tradeoff.png"), dpi=300)
    
    print(f"\n参数影响分析图表已保存至: {output_dir}")

if __name__ == "__main__":
    analyze_gcn_rl_parameters()