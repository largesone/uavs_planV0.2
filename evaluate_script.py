import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.switch_backend('Agg')

def evaluate_score_v3(row):
    """
    根据evaluate.py中的奖励函数设计，计算综合评估分数 (V3版本)。
    
    该评估函数与evaluate.py中的total_reward_score计算逻辑保持一致：
    总评分 = (完成率×5.0 + 目标满足率×3.0 - 资源惩罚×2.0 + 
             同步可行率×1.0 + 负载均衡×0.5 + 资源利用率×0.5 - 
             归一化路径长度×0.1) × 100 - 死锁无人机数×10
    """
    score = 0.0
    
    # 1. 完成率 (completion_rate) - 权重 5.0
    # 任务完成的百分比，非常重要
    completion_score = row['completion_rate'] * 5.0
    score += completion_score
    
    # 2. 目标满足率 (satisfied_targets_rate) - 权重 3.0
    # 达成目标的比率，与完成率类似，但更侧重于目标本身
    satisfied_score = row['satisfied_targets_rate'] * 3.0
    score += satisfied_score
    
    # 3. 资源惩罚 (resource_penalty) - 权重 -2.0 (负向)
    # 未满足资源需求造成的惩罚
    resource_penalty = row['resource_penalty'] * 2.0
    score -= resource_penalty
    
    # 4. 同步可行率 (sync_feasibility_rate) - 权重 1.0
    # 协同任务中同步完成的可能性
    sync_score = row['sync_feasibility_rate'] * 1.0
    score += sync_score
    
    # 5. 负载均衡得分 (load_balance_score) - 权重 0.5
    # 衡量任务分配的均衡性
    balance_score = row['load_balance_score'] * 0.5
    score += balance_score
    
    # 6. 资源利用率 (resource_utilization_rate) - 权重 0.5
    # 资源有效利用的程度
    utilization_score = row['resource_utilization_rate'] * 0.5
    score += utilization_score
    
    # 7. 路径总长度惩罚 - 权重 -0.1 (负向)
    # 路径越长，成本越高，转换为惩罚项
    # 使用相对惩罚，避免绝对值过大
    if row['total_distance'] > 0:
        # 归一化路径长度惩罚，假设最大路径长度为100000
        normalized_distance = min(row['total_distance'] / 100000.0, 1.0)
        distance_penalty = normalized_distance * 0.1
        score -= distance_penalty
    
    # 8. 死锁惩罚 - 权重 -10 (负向)
    # 死锁是严重问题，直接给予高惩罚
    if row['is_deadlocked'] == 1:
        score -= 10 * row['deadlocked_uav_count']
    
    # 9. 规划时间惩罚 - 额外考虑
    # 规划时间越长，效率越低，但权重较小
    if row['planning_time'] > 0:
        time_penalty = min(row['planning_time'] / 100.0, 1.0) * 0.5  # 最大0.5分惩罚
        score -= time_penalty
    
    return score

def calculate_detailed_analysis(df):
    """
    对评估结果进行详细分析，生成各项指标的统计信息
    """
    analysis = {}
    
    # 基础统计
    analysis['total_tests'] = len(df)
    analysis['successful_tests'] = len(df[df['total_reward_score'] > 0])
    analysis['failed_tests'] = len(df[df['total_reward_score'] <= 0])
    
    # 各算法性能统计
    if 'solver' in df.columns:
        solver_stats = df.groupby('solver').agg({
            'total_reward_score': ['mean', 'std', 'min', 'max'],
            'completion_rate': ['mean', 'max'],
            'satisfied_targets_rate': ['mean', 'max'],
            'resource_utilization_rate': ['mean', 'max'],
            'load_balance_score': ['mean', 'max'],
            'total_distance': ['mean', 'min'],
            'planning_time': ['mean', 'min'],
            'is_deadlocked': ['sum']
        }).round(4)
        analysis['solver_performance'] = solver_stats
    
    # 场景难度分析
    if 'scenario' in df.columns:
        scenario_stats = df.groupby('scenario').agg({
            'total_reward_score': ['mean', 'std'],
            'completion_rate': ['mean', 'min'],
            'is_deadlocked': ['sum']
        }).round(4)
        analysis['scenario_difficulty'] = scenario_stats
    
    # 障碍物影响分析
    if 'obstacle_mode' in df.columns:
        obstacle_stats = df.groupby('obstacle_mode').agg({
            'total_reward_score': ['mean', 'std'],
            'completion_rate': ['mean'],
            'total_distance': ['mean'],
            'is_deadlocked': ['sum']
        }).round(4)
        analysis['obstacle_impact'] = obstacle_stats
    
    return analysis

def generate_evaluation_report(df, output_dir):
    """
    生成详细的评估报告
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算新的评估分数
    df['evaluation_score_v3'] = df.apply(evaluate_score_v3, axis=1)
    
    # 详细分析
    analysis = calculate_detailed_analysis(df)
    
    # 生成报告
    report_lines = []
    report_lines.append("="*60)
    report_lines.append("UAV任务规划算法综合评估报告")
    report_lines.append("="*60)
    report_lines.append("")
    
    # 1. 总体统计
    report_lines.append("1. 总体统计")
    report_lines.append("-"*30)
    report_lines.append(f"总测试数: {analysis['total_tests']}")
    report_lines.append(f"成功测试数: {analysis['successful_tests']}")
    report_lines.append(f"失败测试数: {analysis['failed_tests']}")
    report_lines.append(f"成功率: {analysis['successful_tests']/analysis['total_tests']*100:.2f}%")
    report_lines.append("")
    
    # 2. 算法性能排名
    if 'solver_performance' in analysis:
        report_lines.append("2. 算法性能排名")
        report_lines.append("-"*30)
        solver_scores = analysis['solver_performance'][('total_reward_score', 'mean')].sort_values(ascending=False)
        for i, (solver, score) in enumerate(solver_scores.items(), 1):
            report_lines.append(f"{i}. {solver}: {score:.2f}")
        report_lines.append("")
    
    # 3. 单项指标分析
    report_lines.append("3. 单项指标分析")
    report_lines.append("-"*30)
    
    # 完成率分析
    completion_stats = df['completion_rate'].describe()
    report_lines.append("完成率分析:")
    report_lines.append(f"  平均完成率: {completion_stats['mean']:.4f}")
    report_lines.append(f"  最高完成率: {completion_stats['max']:.4f}")
    report_lines.append(f"  最低完成率: {completion_stats['min']:.4f}")
    report_lines.append("")
    
    # 目标满足率分析
    satisfied_stats = df['satisfied_targets_rate'].describe()
    report_lines.append("目标满足率分析:")
    report_lines.append(f"  平均满足率: {satisfied_stats['mean']:.4f}")
    report_lines.append(f"  最高满足率: {satisfied_stats['max']:.4f}")
    report_lines.append(f"  最低满足率: {satisfied_stats['min']:.4f}")
    report_lines.append("")
    
    # 资源利用率分析
    utilization_stats = df['resource_utilization_rate'].describe()
    report_lines.append("资源利用率分析:")
    report_lines.append(f"  平均利用率: {utilization_stats['mean']:.4f}")
    report_lines.append(f"  最高利用率: {utilization_stats['max']:.4f}")
    report_lines.append(f"  最低利用率: {utilization_stats['min']:.4f}")
    report_lines.append("")
    
    # 任务分配均衡性分析
    balance_stats = df['load_balance_score'].describe()
    report_lines.append("任务分配均衡性分析:")
    report_lines.append(f"  平均均衡性: {balance_stats['mean']:.4f}")
    report_lines.append(f"  最高均衡性: {balance_stats['max']:.4f}")
    report_lines.append(f"  最低均衡性: {balance_stats['min']:.4f}")
    report_lines.append("")
    
    # 路径长度分析
    distance_stats = df['total_distance'].describe()
    report_lines.append("路径长度分析:")
    report_lines.append(f"  平均路径长度: {distance_stats['mean']:.2f}")
    report_lines.append(f"  最短路径长度: {distance_stats['min']:.2f}")
    report_lines.append(f"  最长路径长度: {distance_stats['max']:.2f}")
    report_lines.append("")
    
    # 死锁分析
    deadlock_count = df['is_deadlocked'].sum()
    report_lines.append("死锁分析:")
    report_lines.append(f"  死锁发生次数: {deadlock_count}")
    report_lines.append(f"  死锁率: {deadlock_count/len(df)*100:.2f}%")
    report_lines.append("")
    
    # 4. 场景难度分析
    if 'scenario_difficulty' in analysis:
        report_lines.append("4. 场景难度分析")
        report_lines.append("-"*30)
        scenario_scores = analysis['scenario_difficulty'][('total_reward_score', 'mean')].sort_values()
        for scenario, score in scenario_scores.items():
            report_lines.append(f"  {scenario}: {score:.2f}")
        report_lines.append("")
    
    # 5. 障碍物影响分析
    if 'obstacle_impact' in analysis:
        report_lines.append("5. 障碍物影响分析")
        report_lines.append("-"*30)
        obstacle_scores = analysis['obstacle_impact'][('total_reward_score', 'mean')]
        for mode, score in obstacle_scores.items():
            report_lines.append(f"  {mode}: {score:.2f}")
        report_lines.append("")
    
    # 6. 评估分数分布
    report_lines.append("6. 评估分数分布")
    report_lines.append("-"*30)
    score_stats = df['evaluation_score_v3'].describe()
    report_lines.append(f"  平均分数: {score_stats['mean']:.2f}")
    report_lines.append(f"  最高分数: {score_stats['max']:.2f}")
    report_lines.append(f"  最低分数: {score_stats['min']:.2f}")
    report_lines.append(f"  标准差: {score_stats['std']:.2f}")
    report_lines.append("")
    
    # 7. 综合建议
    report_lines.append("7. 综合建议")
    report_lines.append("-"*30)
    
    # 找出最佳算法
    if 'solver_performance' in analysis:
        best_solver = analysis['solver_performance'][('total_reward_score', 'mean')].idxmax()
        report_lines.append(f"  推荐算法: {best_solver}")
    
    # 找出最佳场景
    if 'scenario_difficulty' in analysis:
        best_scenario = analysis['scenario_difficulty'][('total_reward_score', 'mean')].idxmax()
        report_lines.append(f"  最佳场景: {best_scenario}")
    
    # 性能改进建议
    if df['completion_rate'].mean() < 0.8:
        report_lines.append("  建议: 提高算法完成率，可能需要调整资源分配策略")
    
    if df['satisfied_targets_rate'].mean() < 0.8:
        report_lines.append("  建议: 提高目标满足率，可能需要优化任务分配算法")
    
    if df['resource_utilization_rate'].mean() < 0.7:
        report_lines.append("  建议: 提高资源利用率，可能需要改进资源分配机制")
    
    if df['is_deadlocked'].sum() > 0:
        report_lines.append("  建议: 减少死锁发生，可能需要改进协同规划算法")
    
    # 保存报告
    report_path = os.path.join(output_dir, "evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(report_lines))
    
    return report_path

def create_evaluation_visualizations(df, output_dir):
    """
    创建评估结果的可视化图表
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 算法性能对比
    if 'solver' in df.columns:
        plt.figure(figsize=(12, 8))
        
        # 综合评分对比
        plt.subplot(2, 3, 1)
        solver_scores = df.groupby('solver')['evaluation_score_v3'].mean().sort_values(ascending=False)
        solver_scores.plot(kind='bar', color='skyblue')
        plt.title('算法综合评分对比')
        plt.ylabel('平均评分')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 完成率对比
        plt.subplot(2, 3, 2)
        completion_rates = df.groupby('solver')['completion_rate'].mean().sort_values(ascending=False)
        completion_rates.plot(kind='bar', color='lightgreen')
        plt.title('算法完成率对比')
        plt.ylabel('平均完成率')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 目标满足率对比
        plt.subplot(2, 3, 3)
        satisfied_rates = df.groupby('solver')['satisfied_targets_rate'].mean().sort_values(ascending=False)
        satisfied_rates.plot(kind='bar', color='orange')
        plt.title('算法目标满足率对比')
        plt.ylabel('平均满足率')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 资源利用率对比
        plt.subplot(2, 3, 4)
        utilization_rates = df.groupby('solver')['resource_utilization_rate'].mean().sort_values(ascending=False)
        utilization_rates.plot(kind='bar', color='red')
        plt.title('算法资源利用率对比')
        plt.ylabel('平均利用率')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 任务均衡性对比
        plt.subplot(2, 3, 5)
        balance_scores = df.groupby('solver')['load_balance_score'].mean().sort_values(ascending=False)
        balance_scores.plot(kind='bar', color='purple')
        plt.title('算法任务均衡性对比')
        plt.ylabel('平均均衡性')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 路径长度对比
        plt.subplot(2, 3, 6)
        distance_means = df.groupby('solver')['total_distance'].mean().sort_values()
        distance_means.plot(kind='bar', color='brown')
        plt.title('算法平均路径长度对比')
        plt.ylabel('平均路径长度')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'algorithm_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. 评分分布直方图
    plt.figure(figsize=(10, 6))
    plt.hist(df['evaluation_score_v3'], bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.title('评估分数分布')
    plt.xlabel('评估分数')
    plt.ylabel('频次')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'score_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 各项指标相关性热力图
    if 'solver' in df.columns:
        correlation_metrics = ['evaluation_score_v3', 'completion_rate', 'satisfied_targets_rate', 
                             'resource_utilization_rate', 'load_balance_score', 'total_distance']
        
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[correlation_metrics].corr()
        plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
        plt.colorbar()
        plt.xticks(range(len(correlation_metrics)), correlation_metrics, rotation=45)
        plt.yticks(range(len(correlation_metrics)), correlation_metrics)
        
        # 添加相关系数标签
        for i in range(len(correlation_metrics)):
            for j in range(len(correlation_metrics)):
                plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                        ha='center', va='center', color='white' if correlation_matrix.iloc[i, j] < 0 else 'black')
        
        plt.title('指标相关性热力图')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()

def calculate_new_scores_for_csv_v3(input_csv_path, output_csv_path, output_dir="output/evaluation_analysis"):
    """
    读取CSV文件，为每一行计算新的综合评估分数（V3版本），并生成详细分析报告
    
    Args:
        input_csv_path (str): 输入CSV文件的路径
        output_csv_path (str): 输出CSV文件的路径，将包含新的评估分数
        output_dir (str): 分析报告和图表输出目录
    """
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_csv_path}")
        return

    # 检查所需的列是否存在
    required_columns = [
        'total_reward_score', 'completion_rate', 'satisfied_targets_rate',
        'resource_utilization_rate', 'total_distance', 'planning_time',
        'is_deadlocked', 'deadlocked_uav_count', 'resource_penalty',
        'sync_feasibility_rate', 'load_balance_score'
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"错误: CSV文件中缺少以下必要列: {missing_columns}")
        return

    # 应用评估函数到每一行，创建新列 'evaluation_score_v3'
    df['evaluation_score_v3'] = df.apply(evaluate_score_v3, axis=1)

    # 保存到新的CSV文件
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"新的评估分数 (V3) 已计算并保存到 {output_csv_path}")

    # 生成详细分析报告
    report_path = generate_evaluation_report(df, output_dir)
    print(f"详细评估报告已保存到 {report_path}")

    # 创建可视化图表
    create_evaluation_visualizations(df, output_dir)
    print(f"可视化图表已保存到 {output_dir}")

    # 打印简要统计
    print(f"\n评估结果统计:")
    print(f"总测试数: {len(df)}")
    print(f"平均评分: {df['evaluation_score_v3'].mean():.2f}")
    print(f"最高评分: {df['evaluation_score_v3'].max():.2f}")
    print(f"最低评分: {df['evaluation_score_v3'].min():.2f}")
    
    if 'solver' in df.columns:
        best_solver = df.groupby('solver')['evaluation_score_v3'].mean().idxmax()
        print(f"最佳算法: {best_solver}")

# --- 如何使用这个函数 ---
if __name__ == "__main__":
    # 假设您的CSV文件名为 'batch_test_results.csv' 并且与此Python脚本在同一目录下
    input_file = 'test_batch_results.csv'
    output_file = 'test_batch_results_with_evaluation_v3.csv'
    output_dir = 'output/evaluation_analysis'

    # 调用函数计算并保存新的分数
    calculate_new_scores_for_csv_v3(input_file, output_file, output_dir)

    print("完成！")