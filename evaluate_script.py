import pandas as pd
import numpy as np
# 更改评分方案
def new_evaluate_score_v2(row):
    """
    根据给定的CSV行（Pandas Series），计算一个新的综合评估分数 (V2版本)。

    该评估函数旨在更全面地考量UAV任务规划的各个方面，并对不同指标进行适当的缩放和加权。
    得分越高，表示规划方案越好。
    """
    score = 0.0

    # --- 正向贡献指标（越高越好） ---

    # 1. 总奖励得分 (total_reward_score)
    # 这是一个关键指标，直接反映任务完成质量。
    # 考虑到其可能为负值，直接加权。
    score += row['total_reward_score'] * 0.45

    # 2. 完成率 (completion_rate)
    # 任务完成的百分比，非常重要。0-1 范围，放大到0-100。
    score += row['completion_rate'] * 100.0 * 0.20

    # 3. 目标满足率 (satisfied_targets_rate)
    # 达成目标的比率，与完成率类似，但更侧重于目标本身。0-1 范围，放大到0-100。
    score += row['satisfied_targets_rate'] * 100.0 * 0.15

    # 4. 资源利用率 (resource_utilization_rate)
    # 资源有效利用的程度，0-1 范围，放大到0-50。
    score += row['resource_utilization_rate'] * 50.0 * 0.05

    # 5. 同步可行率 (sync_feasibility_rate)
    # 协同任务中同步完成的可能性，0-1 范围，放大到0-20。
    score += row['sync_feasibility_rate'] * 20.0 * 0.05

    # 6. 负载均衡得分 (load_balance_score)
    # 衡量任务分配的均衡性，0-1 范围，放大到0-20。
    score += row['load_balance_score'] * 20.0 * 0.05

    # --- 负向贡献指标（越低越好，转化为惩罚项） ---

    # 7. 总航程 (total_distance)
    # 航程越长，成本越高。转换为惩罚项。
    # 考虑到航程数值可能很大，我们对其进行非线性处理（例如，倒数或对其进行某种标准化）。
    # 这里我们使用一个惩罚系数，并确保不会除以零。
    # 为了避免与正向得分相差过大，可以进行适当缩放。
    if row['total_distance'] > 1: # 避免除以过小的数导致惩罚过大
        score -= (row['total_distance'] / 1000.0) * 0.05 # 每1000米航程扣除0.05分

    # 8. 规划时间 (planning_time)
    # 规划时间越长，效率越低。转换为惩罚项。
    # 同样进行缩放。
    if row['planning_time'] > 0:
        score -= row['planning_time'] * 0.5 # 每秒规划时间扣除0.5分

    # 9. 死锁状态 (is_deadlocked)
    # 死锁是严重问题，直接给予高惩罚。1表示死锁，0表示无死锁。
    if row['is_deadlocked'] == 1:
        score -= 500.0 # 极高惩罚

    # 10. 死锁无人机数量 (deadlocked_uav_count)
    # 死锁的无人机越多，惩罚越大。
    score -= row['deadlocked_uav_count'] * 100.0 # 每死锁一架UAV扣除100分

    # 11. 资源惩罚 (resource_penalty)
    # 未满足资源需求造成的惩罚，0-1 范围，越高越表示问题越大，放大到0-100作为惩罚。
    score -= row['resource_penalty'] * 100.0 * 0.10

    return score

def calculate_new_scores_for_csv_v2(input_csv_path, output_csv_path):
    """
    读取CSV文件，为每一行计算新的综合评估分数（V2版本），并将结果保存到新的CSV文件。

    Args:
        input_csv_path (str): 输入CSV文件的路径。
        output_csv_path (str): 输出CSV文件的路径，将包含新的评估分数。
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

    # 应用评估函数到每一行，创建新列 'new_evaluation_score_v2'
    df['new_evaluation_score_v2'] = df.apply(new_evaluate_score_v2, axis=1)

    # 保存到新的CSV文件
    df.to_csv(output_csv_path, index=False)
    print(f"新的评估分数 (V2) 已计算并保存到 {output_csv_path}")

# --- 如何使用这个函数 ---
# 假设您的CSV文件名为 'batch_test_results.csv' 并且与此Python脚本在同一目录下
input_file = 'batch_test_results-0702.csv'
output_file = 'batch_test_results_with_new_scores_v2.csv'

# 调用函数计算并保存新的分数
calculate_new_scores_for_csv_v2(input_file, output_file)

print("完成！")


# print("请取消注释 `calculate_new_scores_for_csv_v2(input_file, output_file)` 行来运行函数。")
# print("在运行前，请确保您的CSV文件包含函数中假定的所有列名。")

# # 为了演示，我们可以手动创建一个简化的DataFrame并测试评估函数
# print("\n--- 评估函数 V2 示例 ---")
# sample_data = {
#     'total_reward_score': [297.22, 667.59, 707.45, -2593.21, 849.86, -1560.09],
#     'completion_rate': [0.6623, 1.0, 1.0, 0.9409, 1.0, 0.9512],
#     'satisfied_targets_rate': [0.6667, 1.0, 1.0, 0.6667, 1.0, 0.7647],
#     'resource_utilization_rate': [0.6248, 0.9432, 0.9432, 0.8778, 0.9666, 0.8521],
#     'total_distance': [27549.23, 32890.78, 28922.41, 342614.09, 14822.34, 242544.91],
#     'planning_time': [4.76, 2.39, 2.59, 44.04, 2.97, 177.07],
#     'is_deadlocked': [0, 0, 0, 0, 0, 0],
#     'deadlocked_uav_count': [0, 0, 0, 0, 0, 0],
#     'resource_penalty': [0.3377, 0.0, 0.0, 0.0591, 0.0, 0.0488],
#     'sync_feasibility_rate': [0.3333, 1.0, 1.0, 0.8333, 1.0, 0.8077],
#     'load_balance_score': [0.8908, 0.9868, 0.9903, 0.9418, 0.9951, 0.9343]
# }
# sample_df = pd.DataFrame(sample_data)
# sample_df['new_evaluation_score_v2'] = sample_df.apply(new_evaluate_score_v2, axis=1)
# print(sample_df[['total_reward_score', 'completion_rate', 'total_distance', 'is_deadlocked', 'planning_time', 'new_evaluation_score_v2']])