# -*- coding: utf-8 -*-
# 文件名: batch_tester.py
# 描述: 批量测试脚本，集成所有算法对比

import os
import sys
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm

# 导入所有求解器
from main import Config, run_scenario, calculate_economic_sync_speeds, DirectedGraph
from scenarios import get_small_scenario, get_complex_scenario
from GASolver import GASolver
from GreedySolver import GreedySolver
from PSOSolver import ImprovedPSOSolver
from ACOSolver import ImprovedACOSolver
from CBBASolver import ImprovedCBBASolver

def run_batch_test(scenarios: List[Tuple], algorithms: List[str], config: Config) -> pd.DataFrame:
    """运行批量测试"""
    results = []
    
    print(f"开始批量测试，场景数: {len(scenarios)}, 算法数: {len(algorithms)}")
    
    for scenario_idx, (scenario_name, uavs, targets, obstacles) in enumerate(scenarios):
        print(f"\n=== 测试场景 {scenario_idx + 1}: {scenario_name} ===")
        print(f"无人机数量: {len(uavs)}, 目标数量: {len(targets)}")
        
        for algorithm in algorithms:
            print(f"\n--- 测试算法: {algorithm} ---")
            
            try:
                start_time = time.time()
                
                if algorithm == "RL":
                    # RL算法使用run_scenario
                    plan, training_time, deadlocked_tasks = run_scenario(
                        config, uavs, targets, obstacles, scenario_name,
                        save_visualization=False, show_visualization=False, save_report=False
                    )
                    planning_time = 0.0
                    
                elif algorithm == "GA":
                    solver = GASolver(uavs, targets, obstacles, config)
                    plan, training_time, planning_time = solver.solve()
                    deadlocked_tasks = {}
                    
                elif algorithm == "Greedy":
                    graph = DirectedGraph(uavs, targets, n_phi=config.GRAPH_N_PHI)
                    solver = GreedySolver(uavs, targets, graph, config)
                    task_assignments, training_time = solver.solve()
                    plan, deadlocked_tasks = calculate_economic_sync_speeds(
                        task_assignments, uavs, targets, graph, obstacles, config
                    )
                    planning_time = (time.time() - start_time) - training_time
                    
                elif algorithm == "PSO":
                    solver = ImprovedPSOSolver(uavs, targets, obstacles, config)
                    plan, training_time, planning_time = solver.solve()
                    deadlocked_tasks = {}
                    
                elif algorithm == "ACO":
                    solver = ImprovedACOSolver(uavs, targets, obstacles, config)
                    plan, training_time, planning_time = solver.solve()
                    deadlocked_tasks = {}
                    
                elif algorithm == "CBBA":
                    solver = ImprovedCBBASolver(uavs, targets, obstacles, config)
                    plan, training_time, planning_time = solver.solve()
                    deadlocked_tasks = {}
                    
                else:
                    print(f"未知算法: {algorithm}")
                    continue
                
                total_time = time.time() - start_time
                
                # 评估结果
                evaluation = evaluate_plan(plan, uavs, targets, obstacles, config)
                
                # 记录结果
                result = {
                    'scenario': scenario_name,
                    'algorithm': algorithm,
                    'uav_count': len(uavs),
                    'target_count': len(targets),
                    'training_time': training_time,
                    'planning_time': planning_time,
                    'total_time': total_time,
                    'completion_rate': evaluation['completion_rate'],
                    'resource_utilization': evaluation['resource_utilization'],
                    'total_distance': evaluation['total_distance'],
                    'load_balance': evaluation['load_balance'],
                    'deadlocked_tasks': len(deadlocked_tasks) if deadlocked_tasks else 0
                }
                
                results.append(result)
                
                print(f"完成率: {evaluation['completion_rate']:.2%}")
                print(f"资源利用率: {evaluation['resource_utilization']:.2%}")
                print(f"总距离: {evaluation['total_distance']:.2f}")
                print(f"负载均衡: {evaluation['load_balance']:.2f}")
                print(f"耗时: {total_time:.2f}s")
                
            except Exception as e:
                print(f"算法 {algorithm} 在场景 {scenario_name} 中出错: {e}")
                continue
    
    return pd.DataFrame(results)

def evaluate_plan(plan: Dict, uavs: List, targets: List, obstacles: List, config: Config) -> Dict:
    """评估计划质量"""
    if not plan:
        return {
            'completion_rate': 0.0,
            'resource_utilization': 0.0,
            'total_distance': 0.0,
            'load_balance': 0.0
        }
    
    # 计算完成率
    total_demand = sum(np.sum(t.resources) for t in targets)
    total_contribution = 0.0
    
    # 计算资源利用率
    total_initial_resources = sum(np.sum(u.initial_resources) for u in uavs)
    total_used_resources = 0.0
    
    # 计算总距离
    total_distance = 0.0
    
    # 计算负载均衡
    uav_contributions = []
    
    for uav_id, assignments in plan.items():
        uav = next((u for u in uavs if u.id == uav_id), None)
        if not uav:
            continue
            
        uav_contribution = 0.0
        uav_distance = 0.0
        
        for assignment in assignments:
            target_id = assignment['target_id']
            target = next((t for t in targets if t.id == target_id), None)
            
            if target:
                # 计算资源贡献
                resource_cost = assignment.get('resource_cost', target.resources)
                contribution = np.minimum(uav.initial_resources, resource_cost)
                total_contribution += np.sum(contribution)
                total_used_resources += np.sum(contribution)
                uav_contribution += np.sum(contribution)
                
                # 计算距离
                distance = assignment.get('distance', 0.0)
                uav_distance += distance
                total_distance += distance
        
        uav_contributions.append(uav_contribution)
    
    # 计算指标
    completion_rate = total_contribution / total_demand if total_demand > 0 else 0.0
    resource_utilization = total_used_resources / total_initial_resources if total_initial_resources > 0 else 0.0
    
    # 负载均衡（标准差越小越好）
    if uav_contributions:
        load_balance = 1.0 / (1.0 + np.std(uav_contributions))
    else:
        load_balance = 0.0
    
    return {
        'completion_rate': completion_rate,
        'resource_utilization': resource_utilization,
        'total_distance': total_distance,
        'load_balance': load_balance
    }

def generate_comprehensive_report(results_df: pd.DataFrame, output_dir: str = "output"):
    """生成综合报告"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 保存详细结果
    results_df.to_csv(f"{output_dir}/batch_test_results.csv", index=False)
    
    # 2. 生成算法对比图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('算法性能对比', fontsize=16, fontweight='bold')
    
    # 完成率对比
    ax1 = axes[0, 0]
    completion_data = results_df.groupby('algorithm')['completion_rate'].mean()
    completion_data.plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_title('平均完成率')
    ax1.set_ylabel('完成率')
    ax1.tick_params(axis='x', rotation=45)
    
    # 资源利用率对比
    ax2 = axes[0, 1]
    utilization_data = results_df.groupby('algorithm')['resource_utilization'].mean()
    utilization_data.plot(kind='bar', ax=ax2, color='lightgreen')
    ax2.set_title('平均资源利用率')
    ax2.set_ylabel('资源利用率')
    ax2.tick_params(axis='x', rotation=45)
    
    # 总距离对比
    ax3 = axes[1, 0]
    distance_data = results_df.groupby('algorithm')['total_distance'].mean()
    distance_data.plot(kind='bar', ax=ax3, color='lightcoral')
    ax3.set_title('平均总距离')
    ax3.set_ylabel('距离')
    ax3.tick_params(axis='x', rotation=45)
    
    # 负载均衡对比
    ax4 = axes[1, 1]
    balance_data = results_df.groupby('algorithm')['load_balance'].mean()
    balance_data.plot(kind='bar', ax=ax4, color='gold')
    ax4.set_title('平均负载均衡')
    ax4.set_ylabel('负载均衡度')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/algorithm_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 生成详细统计报告
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("无人机任务规划算法批量测试报告")
    report_lines.append("=" * 80)
    report_lines.append(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"测试场景数: {len(results_df['scenario'].unique())}")
    report_lines.append(f"测试算法数: {len(results_df['algorithm'].unique())}")
    report_lines.append(f"总测试次数: {len(results_df)}")
    report_lines.append("")
    
    # 算法性能排名
    report_lines.append("算法性能排名:")
    report_lines.append("-" * 40)
    
    # 按完成率排名
    completion_ranking = results_df.groupby('algorithm')['completion_rate'].mean().sort_values(ascending=False)
    report_lines.append("完成率排名:")
    for i, (algo, rate) in enumerate(completion_ranking.items(), 1):
        report_lines.append(f"{i}. {algo}: {rate:.2%}")
    
    report_lines.append("")
    
    # 按资源利用率排名
    utilization_ranking = results_df.groupby('algorithm')['resource_utilization'].mean().sort_values(ascending=False)
    report_lines.append("资源利用率排名:")
    for i, (algo, rate) in enumerate(utilization_ranking.items(), 1):
        report_lines.append(f"{i}. {algo}: {rate:.2%}")
    
    report_lines.append("")
    
    # 按负载均衡排名
    balance_ranking = results_df.groupby('algorithm')['load_balance'].mean().sort_values(ascending=False)
    report_lines.append("负载均衡排名:")
    for i, (algo, rate) in enumerate(balance_ranking.items(), 1):
        report_lines.append(f"{i}. {algo}: {rate:.3f}")
    
    report_lines.append("")
    
    # 按总时间排名
    time_ranking = results_df.groupby('algorithm')['total_time'].mean().sort_values()
    report_lines.append("平均耗时排名:")
    for i, (algo, time_val) in enumerate(time_ranking.items(), 1):
        report_lines.append(f"{i}. {algo}: {time_val:.2f}s")
    
    report_lines.append("")
    
    # 详细统计
    report_lines.append("详细统计:")
    report_lines.append("-" * 40)
    
    for algorithm in results_df['algorithm'].unique():
        algo_data = results_df[results_df['algorithm'] == algorithm]
        report_lines.append(f"\n{algorithm}算法:")
        report_lines.append(f"  平均完成率: {algo_data['completion_rate'].mean():.2%}")
        report_lines.append(f"  平均资源利用率: {algo_data['resource_utilization'].mean():.2%}")
        report_lines.append(f"  平均负载均衡: {algo_data['load_balance'].mean():.3f}")
        report_lines.append(f"  平均总距离: {algo_data['total_distance'].mean():.2f}")
        report_lines.append(f"  平均耗时: {algo_data['total_time'].mean():.2f}s")
        report_lines.append(f"  测试次数: {len(algo_data)}")
    
    # 保存报告
    with open(f"{output_dir}/batch_test_report.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"综合报告已生成:")
    print(f"  - 详细结果: {output_dir}/batch_test_results.csv")
    print(f"  - 对比图表: {output_dir}/algorithm_comparison.png")
    print(f"  - 统计报告: {output_dir}/batch_test_report.txt")

def main():
    """主函数"""
    print("=== 无人机任务规划算法批量测试 ===")
    
    # 配置
    config = Config()
    
    # 定义测试场景
    scenarios = []
    
    # 小场景
    uavs, targets, obstacles = get_small_scenario(config.OBSTACLE_TOLERANCE)
    scenarios.append(("小场景测试", uavs, targets, obstacles))
    
    # 复杂场景
    uavs, targets, obstacles = get_complex_scenario(config.OBSTACLE_TOLERANCE)
    scenarios.append(("复杂场景测试", uavs, targets, obstacles))
    
    # 定义测试算法
    algorithms = ["RL", "GA", "Greedy", "PSO", "ACO", "CBBA"]
    
    # 运行批量测试
    results_df = run_batch_test(scenarios, algorithms, config)
    
    # 生成报告
    generate_comprehensive_report(results_df)
    
    print("\n=== 批量测试完成 ===")
    
    # 检查资源利用率
    avg_utilization = results_df['resource_utilization'].mean()
    print(f"平均资源利用率: {avg_utilization:.2%}")
    
    if avg_utilization >= 0.8:
        print("✓ 资源利用率达到80%以上目标")
    else:
        print("✗ 资源利用率未达到80%目标，需要进一步优化")

if __name__ == "__main__":
    main()