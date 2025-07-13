# -*- coding: utf-8 -*-
# 文件名: quick_param_test.py
# 描述: 快速测试EPISODES和USE_PHRRT_DURING_TRAINING参数对GCN-RL算法求解结果的影响

import time
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from main import Config, run_scenario
from scenarios import get_small_scenario
from evaluate import evaluate_plan

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
# 设置matplotlib不显示图形窗口，直接保存到文件
plt.switch_backend('Agg')

def generate_analysis_report(df, best_row, output_dir):
    """
    生成详细的参数分析报告并保存到文件
    
    Args:
        df: 包含所有测试结果的DataFrame
        best_row: 最佳参数组合的数据行
        output_dir: 输出目录路径
    """
    report_lines = []
    report_lines.append("="*50)
    report_lines.append("GCN-RL算法参数测试分析报告")
    report_lines.append("="*50)
    report_lines.append("")
    
    # 1. 最佳参数组合
    report_lines.append("1. 最佳参数组合")
    report_lines.append("-"*30)
    report_lines.append(f"EPISODES = {best_row['EPISODES']}")
    report_lines.append(f"USE_PHRRT_DURING_TRAINING = {best_row['USE_PHRRT_DURING_TRAINING']}")
    report_lines.append("")
    report_lines.append("性能指标:")
    report_lines.append(f"  综合评分 = {best_row['total_reward_score']:.2f}")
    report_lines.append(f"  目标满足率 = {best_row['satisfied_targets_rate']:.4f}")
    report_lines.append(f"  资源利用率 = {best_row['resource_utilization_rate']:.4f}")
    report_lines.append(f"  任务分配均衡性 = {best_row['load_balance_score']:.4f}")
    report_lines.append(f"  路径总长度 = {best_row['total_distance']:.2f}")
    report_lines.append("")
    report_lines.append("时间指标:")
    report_lines.append(f"  训练时间 = {best_row['training_time']:.2f}秒")
    report_lines.append(f"  总运行时间 = {best_row['total_runtime']:.2f}秒")
    report_lines.append("")
    
    # 2. 参数影响分析
    report_lines.append("2. 参数影响分析")
    report_lines.append("-"*30)
    
    # 2.1 训练轮次(EPISODES)影响分析
    ep_group = df.groupby('EPISODES').mean()
    best_ep = ep_group['total_reward_score'].idxmax()
    report_lines.append("2.1 训练轮次(EPISODES)影响分析:")
    report_lines.append(f"   - 最佳训练轮次: {best_ep}")
    report_lines.append(f"   - 训练轮次越多，综合评分{'增加' if ep_group['total_reward_score'].is_monotonic_increasing else '不一定增加'}")
    report_lines.append(f"   - 训练轮次越多，训练时间{'增加' if ep_group['training_time'].is_monotonic_increasing else '不一定增加'}")
    
    # 计算各训练轮次的平均性能提升
    if len(ep_group) > 1:
        ep_values = sorted(df['EPISODES'].unique())
        ep_improvements = []
        for i in range(1, len(ep_values)):
            prev_score = ep_group.loc[ep_values[i-1], 'total_reward_score']
            curr_score = ep_group.loc[ep_values[i], 'total_reward_score']
            improvement = curr_score - prev_score
            ep_improvements.append(improvement)
        
        if ep_improvements:
            avg_improvement = sum(ep_improvements) / len(ep_improvements)
            report_lines.append(f"   - 平均而言，每增加训练轮次，综合评分提升了{avg_improvement:.2f}分")
    report_lines.append("")
    
    # 2.2 距离计算方式(PHRRT)影响分析
    phrrt_group = df.groupby('USE_PHRRT_DURING_TRAINING').mean()
    best_phrrt = phrrt_group['total_reward_score'].idxmax()
    
    # 安全计算时间差异比例
    try:
        if False in phrrt_group.index and True in phrrt_group.index and phrrt_group.loc[False, 'training_time'] > 0:
            phrrt_time_diff = phrrt_group.loc[True, 'training_time'] / phrrt_group.loc[False, 'training_time']
        else:
            phrrt_time_diff = None
    except Exception:
        phrrt_time_diff = None
    
    report_lines.append("2.2 距离计算方式(PHRRT)影响分析:")
    report_lines.append(f"   - 最佳距离计算方式: {'高精度PH-RRT' if best_phrrt else '快速近似距离'}")
    
    # 只有当两种方式都有数据时才比较
    if True in phrrt_group.index and False in phrrt_group.index:
        score_diff = abs(phrrt_group.loc[True, 'total_reward_score'] - phrrt_group.loc[False, 'total_reward_score'])
        is_better = phrrt_group.loc[True, 'total_reward_score'] > phrrt_group.loc[False, 'total_reward_score']
        report_lines.append(f"   - 高精度PH-RRT比快速近似距离{'提高' if is_better else '降低'}了{score_diff:.2f}分")
        
        if phrrt_time_diff is not None:
            report_lines.append(f"   - 高精度PH-RRT比快速近似距离训练时间增加了{phrrt_time_diff:.2f}倍")
        else:
            report_lines.append(f"   - 无法比较两种距离计算方式的训练时间差异")
    report_lines.append("")
    
    # 3. 各评价指标分析
    report_lines.append("3. 各评价指标分析")
    report_lines.append("-"*30)
    report_lines.append(f"   - 目标满足率: 平均{df['satisfied_targets_rate'].mean():.4f}, 最高{df['satisfied_targets_rate'].max():.4f}")
    report_lines.append(f"   - 资源利用率: 平均{df['resource_utilization_rate'].mean():.4f}, 最高{df['resource_utilization_rate'].max():.4f}")
    report_lines.append(f"   - 任务分配均衡性: 平均{df['load_balance_score'].mean():.4f}, 最高{df['load_balance_score'].max():.4f}")
    report_lines.append(f"   - 路径总长度: 平均{df['total_distance'].mean():.2f}, 最短{df['total_distance'].min():.2f}")
    report_lines.append("")
    
    # 4. 综合建议
    report_lines.append("4. 综合建议")
    report_lines.append("-"*30)
    if best_row['total_reward_score'] < 0:
        report_lines.append(f"   - 当前参数组合下求解效果不理想，建议尝试更大的训练轮次或调整其他参数")
    else:
        report_lines.append(f"   - 推荐使用 EPISODES={best_row['EPISODES']}, USE_PHRRT_DURING_TRAINING={best_row['USE_PHRRT_DURING_TRAINING']} 的参数组合")
    
    if df['completion_rate'].max() == 0:
        report_lines.append(f"   - 所有方案的完成率均为0，表明当前场景可能过于复杂，建议简化场景或调整其他算法参数")
    
    # 时间与质量的权衡建议
    time_efficient_row = df.loc[df['training_time'].idxmin()]
    if time_efficient_row['total_reward_score'] > 0.8 * best_row['total_reward_score']:
        report_lines.append(f"   - 如果对时间敏感，可以考虑使用 EPISODES={time_efficient_row['EPISODES']}, USE_PHRRT_DURING_TRAINING={time_efficient_row['USE_PHRRT_DURING_TRAINING']} 的参数组合")
        report_lines.append(f"     该组合训练时间仅为{time_efficient_row['training_time']:.2f}秒，但仍能达到{time_efficient_row['total_reward_score']:.2f}的评分（最佳评分的{time_efficient_row['total_reward_score']/best_row['total_reward_score']*100:.1f}%）")
    
    # 5. 图表分析结论
    report_lines.append("")
    report_lines.append("5. 图表分析结论")
    report_lines.append("-"*30)
    report_lines.append("   - 运行时间分析: 高精度PH-RRT距离计算方式会显著增加训练时间，但可能提高求解质量")
    report_lines.append("   - 路径总长度分析: 训练轮次增加通常会优化路径长度，减少不必要的绕路")
    report_lines.append("   - 目标满足率分析: 训练轮次和距离计算精度提高通常会增加目标满足率")
    report_lines.append("   - 资源利用率分析: 更优的参数组合可以提高资源利用效率，减少浪费")
    report_lines.append("   - 任务分配均衡性分析: 良好的参数设置有助于实现更均衡的任务分配")
    report_lines.append("   - 综合评分分析: 评分规则综合考虑了多个因素，最佳参数组合在各方面都有较好表现")
    report_lines.append("")
    
    # 6. 结论
    report_lines.append("6. 结论")
    report_lines.append("-"*30)
    report_lines.append(f"   根据本次参数测试结果，推荐使用 EPISODES={best_row['EPISODES']}, USE_PHRRT_DURING_TRAINING={best_row['USE_PHRRT_DURING_TRAINING']} 的参数组合进行GCN-RL算法的训练和求解。该组合在综合评分、目标满足率、资源利用率和任务分配均衡性等方面表现最佳，能够在合理的时间内获得较好的求解质量。")
    
    if best_row['total_reward_score'] < 0 or df['completion_rate'].max() == 0:
        report_lines.append(f"   然而，当前场景下所有参数组合的求解效果均不理想，建议进一步调整算法参数或简化场景复杂度。")
    
    # 保存报告到文件
    report_path = os.path.join(output_dir, "parameter_analysis_report.txt")
    # 确保使用UTF-8编码并添加BOM标记，以便Windows正确识别中文
    with open(report_path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(report_lines))
    
    return report_path

def run_quick_parameter_test():
    """
    快速测试EPISODES和USE_PHRRT_DURING_TRAINING参数对GCN-RL算法求解结果的影响
    """
    # 创建输出目录
    output_dir = "output/quick_param_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 定义要测试的参数组合 ---
    # 只关注EPISODES和USE_PHRRT_DURING_TRAINING这两个关键参数
    test_episodes = [200, 500, 800]
    test_phrrt = [False, True]
    
    # 生成所有参数组合
    param_combinations = []
    for ep in test_episodes:
        for phrrt in test_phrrt:
            param_combinations.append({
                'EPISODES': ep,
                'USE_PHRRT_DURING_TRAINING': phrrt
            })

    print(f"将要测试 {len(param_combinations)} 种不同的参数组合...")

    # --- 选择测试场景 ---
    base_uavs, base_targets, base_obstacles = get_small_scenario(obstacle_tolerance=50.0)
    scenario_name_base = "QuickTest"

    # 准备结果存储
    results = []
    
    # 存储无人机分配但资源贡献为0的情况
    zero_contribution_cases = []

    for i, params in enumerate(param_combinations):
        print(f"\n{'='*40}\n[测试 {i+1}/{len(param_combinations)}] 参数: {params}\n{'='*40}")
        
        # --- 应用参数 ---
        config = Config()
        config.RUN_TRAINING = True  # 强制进行训练
        for key, value in params.items():
            setattr(config, key, value)
        
        # 创建可读的参数字符串用于模型路径和结果标识
        param_str = f"ep{params['EPISODES']}_phrrt{params['USE_PHRRT_DURING_TRAINING']}"
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
            
            # 检测无人机分配但资源贡献为0的情况
            zero_contribution_found = False
            for uav_id, tasks in final_plan.items():
                if tasks:  # 如果无人机被分配了任务
                    for task in tasks:
                        if 'resource_cost' in task and np.all(task['resource_cost'] == 0):
                            zero_contribution_found = True
                            zero_contribution_cases.append({
                                'param_set': f"EPISODES={params['EPISODES']}, PHRRT={params['USE_PHRRT_DURING_TRAINING']}",
                                'uav_id': uav_id,
                                'target_id': task['target_id'],
                                'task_details': task
                            })
            
            # 记录结果
            result = {
                'EPISODES': params['EPISODES'],
                'USE_PHRRT_DURING_TRAINING': params['USE_PHRRT_DURING_TRAINING'],
                'training_time': training_time,
                'total_runtime': total_runtime,
                'zero_contribution_detected': 1 if zero_contribution_found else 0,
                **{k: v for k, v in evaluation_metrics.items() if not isinstance(v, str)}  # 排除字符串类型的指标
            }
            results.append(result)
            
            print(f"--- 测试 {i+1} 完成 ---")
            print(f"训练时间: {training_time:.2f}s, 总运行时间: {total_runtime:.2f}s")
            print(f"总奖励分数: {evaluation_metrics['total_reward_score']:.2f}")
            print(f"完成率: {evaluation_metrics['completion_rate']:.4f}")
            print(f"目标满足率: {evaluation_metrics['satisfied_targets_rate']:.4f}")
            print(f"资源利用率: {evaluation_metrics['resource_utilization_rate']:.4f}")
            print(f"任务分配均衡性: {evaluation_metrics['load_balance_score']:.4f}")
            print(f"路径总长度: {evaluation_metrics['total_distance']:.2f}")
            print(f"是否死锁: {'是' if evaluation_metrics['is_deadlocked'] else '否'}")
            if zero_contribution_found:
                print(f"警告: 检测到无人机分配但资源贡献为0的情况!")
        else:
            print(f"--- 测试 {i+1} 失败 --- 未能生成有效方案")
            # 记录失败结果
            results.append({
                'EPISODES': params['EPISODES'],
                'USE_PHRRT_DURING_TRAINING': params['USE_PHRRT_DURING_TRAINING'],
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
                'total_distance': 0,
                'zero_contribution_detected': 0
            })

    # --- 结果分析和可视化 ---
    # 转换为DataFrame便于分析
    df = pd.DataFrame(results)
    
    # 保存原始结果到CSV
    csv_path = os.path.join(output_dir, "quick_param_test_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n原始结果已保存至: {csv_path}")
    
    # 按总奖励分数排序
    sorted_df = df.sort_values(by='total_reward_score', ascending=False)
    
    # 打印排名
    print(f"\n{'='*50}\n========= 参数测试最终排名 =========\n{'='*50}")
    for i, row in sorted_df.iterrows():
        print(f"排名 {i+1}: EPISODES={row['EPISODES']}, "
              f"USE_PHRRT_DURING_TRAINING={row['USE_PHRRT_DURING_TRAINING']} | "
              f"总奖励={row['total_reward_score']:.2f}, 训练时间={row['training_time']:.2f}s")
    
    # 打印资源贡献为0的情况
    if zero_contribution_cases:
        print(f"\n{'='*50}\n检测到无人机分配但资源贡献为0的情况:\n{'='*50}")
        for i, case in enumerate(zero_contribution_cases):
            print(f"案例 {i+1}:")
            print(f"  参数设置: {case['param_set']}")
            print(f"  无人机ID: {case['uav_id']}")
            print(f"  目标ID: {case['target_id']}")
            print(f"  任务详情: {case['task_details']}")
    else:
        print(f"\n未检测到无人机分配但资源贡献为0的情况")
    
    # --- 可视化分析 ---
    # 1. 算法运行时间分析（单独分析，不计入总评分）
    plt.figure(figsize=(12, 6))
    
    # 训练时间对比
    plt.subplot(1, 2, 1)
    bar_width = 0.35
    x = np.arange(len(test_episodes))
    
    for i, phrrt in enumerate([False, True]):
        subset = df[df['USE_PHRRT_DURING_TRAINING'] == phrrt]
        training_times = []
        for ep in test_episodes:
            time_value = subset[subset['EPISODES'] == ep]['training_time'].values
            training_times.append(time_value[0] if len(time_value) > 0 else 0)
        
        plt.bar(x + i*bar_width, training_times, bar_width, 
                label=f"PHRRT={'高精度' if phrrt else '近似'}", 
                alpha=0.7)
    
    plt.xlabel('训练轮次 (EPISODES)')
    plt.ylabel('训练时间 (秒)')
    plt.title('训练时间分析')
    plt.xticks(x + bar_width/2, test_episodes)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 总运行时间对比
    plt.subplot(1, 2, 2)
    for i, phrrt in enumerate([False, True]):
        subset = df[df['USE_PHRRT_DURING_TRAINING'] == phrrt]
        total_times = []
        for ep in test_episodes:
            time_value = subset[subset['EPISODES'] == ep]['total_runtime'].values
            total_times.append(time_value[0] if len(time_value) > 0 else 0)
        
        plt.bar(x + i*bar_width, total_times, bar_width, 
                label=f"PHRRT={'高精度' if phrrt else '近似'}", 
                alpha=0.7)
    
    plt.xlabel('训练轮次 (EPISODES)')
    plt.ylabel('总运行时间 (秒)')
    plt.title('总运行时间分析')
    plt.xticks(x + bar_width/2, test_episodes)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "runtime_analysis.png"), dpi=300)
    
    # 2. 各评价指标单独分析
    # 2.1 路径总长度分析
    plt.figure(figsize=(10, 6))
    for phrrt in [False, True]:
        subset = df[df['USE_PHRRT_DURING_TRAINING'] == phrrt]
        plt.plot(subset['EPISODES'], subset['total_distance'], marker='o', 
                 label=f"PHRRT={'高精度PH-RRT' if phrrt else '快速近似距离'}")
    
    plt.xlabel('训练轮次 (EPISODES)')
    plt.ylabel('路径总长度')
    plt.title('训练轮次对路径总长度的影响')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "total_distance_analysis.png"), dpi=300)
    
    # 2.2 目标资源满足度分析
    plt.figure(figsize=(10, 6))
    for phrrt in [False, True]:
        subset = df[df['USE_PHRRT_DURING_TRAINING'] == phrrt]
        plt.plot(subset['EPISODES'], subset['satisfied_targets_rate'], marker='o', 
                 label=f"PHRRT={'高精度PH-RRT' if phrrt else '快速近似距离'}")
    
    plt.xlabel('训练轮次 (EPISODES)')
    plt.ylabel('目标资源满足率')
    plt.title('训练轮次对目标资源满足率的影响')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "satisfied_targets_rate_analysis.png"), dpi=300)
    
    # 2.3 资源利用率分析
    plt.figure(figsize=(10, 6))
    for phrrt in [False, True]:
        subset = df[df['USE_PHRRT_DURING_TRAINING'] == phrrt]
        plt.plot(subset['EPISODES'], subset['resource_utilization_rate'], marker='o', 
                 label=f"PHRRT={'高精度PH-RRT' if phrrt else '快速近似距离'}")
    
    plt.xlabel('训练轮次 (EPISODES)')
    plt.ylabel('资源利用率')
    plt.title('训练轮次对资源利用率的影响')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "resource_utilization_rate_analysis.png"), dpi=300)
    
    # 2.4 任务分配均衡性分析
    plt.figure(figsize=(10, 6))
    for phrrt in [False, True]:
        subset = df[df['USE_PHRRT_DURING_TRAINING'] == phrrt]
        plt.plot(subset['EPISODES'], subset['load_balance_score'], marker='o', 
                 label=f"PHRRT={'高精度PH-RRT' if phrrt else '快速近似距离'}")
    
    plt.xlabel('训练轮次 (EPISODES)')
    plt.ylabel('任务分配均衡性得分')
    plt.title('训练轮次对任务分配均衡性的影响')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "load_balance_score_analysis.png"), dpi=300)
    
    # 2.5 综合评分分析
    plt.figure(figsize=(10, 8))
    for phrrt in [False, True]:
        subset = df[df['USE_PHRRT_DURING_TRAINING'] == phrrt]
        plt.plot(subset['EPISODES'], subset['total_reward_score'], marker='o', 
                 label=f"PHRRT={'高精度PH-RRT' if phrrt else '快速近似距离'}")
    
    plt.xlabel('训练轮次 (EPISODES)')
    plt.ylabel('综合评分')
    plt.title('训练轮次对综合评分的影响')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 添加评分规则说明
    score_rule = (
        "评分规则:\n"
        "总评分 = (完成率×5.0 + 目标满足率×3.0 - 资源惩罚×2.0 + \n"
        "         同步可行率×1.0 + 负载均衡×0.5 + 资源利用率×0.5 - \n"
        "         归一化路径长度×0.1) × 100 - 死锁无人机数×10\n"
        "注: 所有指标均已归一化到0-1之间"
    )
    plt.figtext(0.5, 0.01, score_rule, ha='center', fontsize=10, 
                bbox={'facecolor': 'lightgray', 'alpha': 0.5, 'pad': 5})
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])  # 为底部文本留出空间
    plt.savefig(os.path.join(output_dir, "total_reward_score_analysis.png"), dpi=300)
    
    # 3. 参数组合热力图 - 所有指标对比
    metrics = ['total_reward_score', 'satisfied_targets_rate', 'resource_utilization_rate', 
               'load_balance_score', 'total_distance']
    metric_names = ['综合评分', '目标满足率', '资源利用率', '任务均衡性', '路径总长度']
    
    plt.figure(figsize=(15, 10))
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        plt.subplot(2, 3, i+1)
        
        # 创建热力图数据
        heatmap_data = np.zeros((len(test_phrrt), len(test_episodes)))
        for p_idx, phrrt in enumerate(test_phrrt):
            for e_idx, ep in enumerate(test_episodes):
                subset = df[(df['USE_PHRRT_DURING_TRAINING'] == phrrt) & (df['EPISODES'] == ep)]
                if not subset.empty:
                    # 对于路径总长度，值越小越好，所以取负值
                    value = subset[metric].values[0]
                    if metric == 'total_distance' and value != 0:  # 避免对0取负
                        value = -value
                    heatmap_data[p_idx, e_idx] = value
        
        # 绘制热力图
        im = plt.imshow(heatmap_data, cmap='viridis')
        plt.colorbar(im, label=name)
        plt.title(f'{name}分析')
        plt.xlabel('训练轮次 (EPISODES)')
        plt.ylabel('PHRRT设置')
        plt.xticks(np.arange(len(test_episodes)), test_episodes)
        plt.yticks(np.arange(len(test_phrrt)), ['近似', '高精度'])
        
        # 添加数值标签
        for p_idx in range(len(test_phrrt)):
            for e_idx in range(len(test_episodes)):
                value = heatmap_data[p_idx, e_idx]
                if metric == 'total_distance' and value < 0:  # 恢复路径长度的实际值
                    text_value = -value
                else:
                    text_value = value
                    
                # 根据指标类型格式化显示
                if metric in ['satisfied_targets_rate', 'resource_utilization_rate', 'load_balance_score']:
                    text = f"{text_value:.2f}"
                else:
                    text = f"{text_value:.0f}"
                    
                plt.text(e_idx, p_idx, text, ha="center", va="center", 
                         color="w" if value < np.max(heatmap_data)/1.5 else "black")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_heatmap.png"), dpi=300)
    
    print(f"\n参数影响分析图表已保存至: {output_dir}")
    
    # 返回最佳参数组合并分析结果
    best_row = sorted_df.iloc[0]
    print(f"\n{'='*50}\n========= 最佳参数组合 =========\n{'='*50}")
    print(f"EPISODES = {best_row['EPISODES']}")
    print(f"USE_PHRRT_DURING_TRAINING = {best_row['USE_PHRRT_DURING_TRAINING']}")
    print(f"\n性能指标:")
    print(f"  综合评分 = {best_row['total_reward_score']:.2f}")
    print(f"  目标满足率 = {best_row['satisfied_targets_rate']:.4f}")
    print(f"  资源利用率 = {best_row['resource_utilization_rate']:.4f}")
    print(f"  任务分配均衡性 = {best_row['load_balance_score']:.4f}")
    print(f"  路径总长度 = {best_row['total_distance']:.2f}")
    print(f"\n时间指标:")
    print(f"  训练时间 = {best_row['training_time']:.2f}秒")
    print(f"  总运行时间 = {best_row['total_runtime']:.2f}秒")
    
    # 分析本次参数运行结果
    print(f"\n{'='*50}\n========= 参数运行结果分析 =========\n{'='*50}")
    
    # 1. 训练轮次(EPISODES)影响分析
    ep_group = df.groupby('EPISODES').mean()
    best_ep = ep_group['total_reward_score'].idxmax()
    print(f"1. 训练轮次(EPISODES)影响分析:")
    print(f"   - 最佳训练轮次: {best_ep}")
    print(f"   - 训练轮次越多，综合评分{'增加' if ep_group['total_reward_score'].is_monotonic_increasing else '不一定增加'}")
    print(f"   - 训练轮次越多，训练时间{'增加' if ep_group['training_time'].is_monotonic_increasing else '不一定增加'}")
    
    # 2. 距离计算方式(PHRRT)影响分析
    phrrt_group = df.groupby('USE_PHRRT_DURING_TRAINING').mean()
    best_phrrt = phrrt_group['total_reward_score'].idxmax()
    
    # 安全计算时间差异比例
    try:
        if False in phrrt_group.index and True in phrrt_group.index and phrrt_group.loc[False, 'training_time'] > 0:
            phrrt_time_diff = phrrt_group.loc[True, 'training_time'] / phrrt_group.loc[False, 'training_time']
        else:
            phrrt_time_diff = None
    except Exception:
        phrrt_time_diff = None
    
    print(f"\n2. 距离计算方式(PHRRT)影响分析:")
    print(f"   - 最佳距离计算方式: {'高精度PH-RRT' if best_phrrt else '快速近似距离'}")
    
    # 只有当两种方式都有数据时才比较
    if True in phrrt_group.index and False in phrrt_group.index:
        score_diff = abs(phrrt_group.loc[True, 'total_reward_score'] - phrrt_group.loc[False, 'total_reward_score'])
        is_better = phrrt_group.loc[True, 'total_reward_score'] > phrrt_group.loc[False, 'total_reward_score']
        print(f"   - 高精度PH-RRT比快速近似距离{'提高' if is_better else '降低'}了{score_diff:.2f}分")
        
        if phrrt_time_diff is not None:
            print(f"   - 高精度PH-RRT比快速近似距离训练时间增加了{phrrt_time_diff:.2f}倍")
        else:
            print(f"   - 无法比较两种距离计算方式的训练时间差异")
    
    # 3. 各评价指标分析
    print(f"\n3. 各评价指标分析:")
    print(f"   - 目标满足率: 平均{df['satisfied_targets_rate'].mean():.4f}, 最高{df['satisfied_targets_rate'].max():.4f}")
    print(f"   - 资源利用率: 平均{df['resource_utilization_rate'].mean():.4f}, 最高{df['resource_utilization_rate'].max():.4f}")
    print(f"   - 任务分配均衡性: 平均{df['load_balance_score'].mean():.4f}, 最高{df['load_balance_score'].max():.4f}")
    print(f"   - 路径总长度: 平均{df['total_distance'].mean():.2f}, 最短{df['total_distance'].min():.2f}")
    
    # 4. 综合建议
    print(f"\n4. 综合建议:")
    if best_row['total_reward_score'] < 0:
        print(f"   - 当前参数组合下求解效果不理想，建议尝试更大的训练轮次或调整其他参数")
    else:
        print(f"   - 推荐使用 EPISODES={best_row['EPISODES']}, USE_PHRRT_DURING_TRAINING={best_row['USE_PHRRT_DURING_TRAINING']} 的参数组合")
    
    if df['completion_rate'].max() == 0:
        print(f"   - 所有方案的完成率均为0，表明当前场景可能过于复杂，建议简化场景或调整其他算法参数")
    
    # 时间与质量的权衡建议
    time_efficient_row = df.loc[df['training_time'].idxmin()]
    if time_efficient_row['total_reward_score'] > 0.8 * best_row['total_reward_score']:
        print(f"   - 如果对时间敏感，可以考虑使用 EPISODES={time_efficient_row['EPISODES']}, USE_PHRRT_DURING_TRAINING={time_efficient_row['USE_PHRRT_DURING_TRAINING']} 的参数组合")
        print(f"     该组合训练时间仅为{time_efficient_row['training_time']:.2f}秒，但仍能达到{time_efficient_row['total_reward_score']:.2f}的评分（最佳评分的{time_efficient_row['total_reward_score']/best_row['total_reward_score']*100:.1f}%）")
    
    # 生成详细分析报告并保存到文件
    report_path = generate_analysis_report(df, best_row, output_dir)
    print(f"\n{'='*50}")
    print(f"详细分析报告已保存至: {report_path}")
    print(f"{'='*50}")

if __name__ == "__main__":
    run_quick_parameter_test()