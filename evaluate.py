# -*- coding: utf-8 -*-
# 文件名: evaluate.py
# 描述: 包含用于评估任务规划方案质量的函数。

import numpy as np
from collections import defaultdict

# 全局变量存储各指标最优值
GLOBAL_BEST_METRICS = {
    'completion_rate': 0,
    'satisfied_targets_rate': 0,
    'sync_feasibility_rate': 0,
    'load_balance_score': 0,
    'resource_utilization_rate': 0,
    'total_distance': float('inf')  # 距离越小越好
}

def evaluate_plan(final_plan, uavs, targets, deadlocked_tasks=None) -> dict:
    """评估一个最终方案的综合质量，返回包含各项指标的字典。"""
    # # 预处理任务数据补充resource_cost
    # for uav_id, tasks in final_plan.items():
    #     for task in tasks:
    #         if 'resource_cost' not in task:
    #             target = next((t for t in targets if t.id == task['target_id']), None)
    #             task['resource_cost'] = target.resources.copy() if target else np.zeros_like(uavs[0].initial_resources)

    is_deadlocked = 1 if deadlocked_tasks and any(deadlocked_tasks.values()) else 0
    deadlocked_uav_count = len([uav for uav, tasks in (deadlocked_tasks or {}).items() if tasks])

    if not final_plan or not any(final_plan.values()):
        return {
            'total_reward_score': -2000 if is_deadlocked else -1000, 'is_deadlocked': is_deadlocked,
            'deadlocked_uav_count': deadlocked_uav_count, 'completion_rate': 0,
            'satisfied_targets_count': 0, 
            'total_targets': len(targets),
            'satisfied_targets_rate': 0, 'resource_penalty': 1,
            'sync_feasibility_rate': 0, 'load_balance_score': 0, 'total_distance': 0,
            'resource_utilization_rate': 0, 'total_demand': '[]', 'total_contribution': '[]'
        }

    total_demand = np.sum([t.resources for t in targets], axis=0)
    total_demand_safe = np.maximum(total_demand, 1e-6)

    target_contributions = defaultdict(lambda: np.zeros_like(total_demand, dtype=float))
    all_tasks = [task for tasks in final_plan.values() for task in tasks]
    for task in all_tasks:
        target_id = task['target_id']
        target = next((t for t in targets if t.id == target_id), None)
        if target:
            resource_cost = task.get('resource_cost', target.resources) # 如果没有，则默认认为等于需求
        else:
            resource_cost = task.get('resource_cost', np.zeros(len(total_demand)))
        target_contributions[task['target_id']] += resource_cost

    total_contribution = np.sum(list(target_contributions.values()), axis=0)
    
    satisfied_targets_count = sum(1 for t in targets if np.all(target_contributions[t.id] >= t.resources - 1e-5))
    if not targets:
        satisfied_targets_rate = 1.0
    else:
        raw_rate = satisfied_targets_count / len(targets)
        if raw_rate == 1.0:
            satisfied_targets_rate = 1.0  # 全部完成，得满分
        else:
            # 部分完成时，使用平方来放大差距，鼓励完成更多目标
            satisfied_targets_rate = raw_rate ** 2

    completion_rate = np.mean(np.minimum(total_contribution, total_demand) / total_demand_safe)
    resource_penalty = np.mean(np.maximum(0, total_demand - total_contribution) / total_demand_safe)
    
    if not all_tasks:
        return {
            'total_reward_score': -500, 'is_deadlocked': is_deadlocked, 'deadlocked_uav_count': deadlocked_uav_count,
            'completion_rate': round(completion_rate, 4), 
            'satisfied_targets_count': satisfied_targets_count,
            'total_targets': len(targets),
            'satisfied_targets_rate': round(satisfied_targets_rate, 4), 'resource_penalty': round(resource_penalty, 4),
            'sync_feasibility_rate': 0, 'load_balance_score': 0, 'total_distance': 0, 'resource_utilization_rate': 0,
            'total_demand': np.array2string(total_demand, formatter={'float_kind': lambda x: "%.0f" % x}),
            'total_contribution': np.array2string(total_contribution, formatter={'float_kind': lambda x: "%.1f" % x}),
        }
    
    sync_feasibility_rate = sum(1 for task in all_tasks if task['is_sync_feasible']) / len(all_tasks)
    total_distance = sum(task['distance'] for task in all_tasks if 'distance' in task)

    uav_expenditure = defaultdict(lambda: np.zeros(len(uavs[0].initial_resources)))
    for uav_id, tasks in final_plan.items():
        for task in tasks:
            uav_expenditure[uav_id] += task.get('resource_cost', np.zeros(len(uavs[0].initial_resources)))
    
    load_ratios = [np.mean(uav_expenditure[u.id] / np.maximum(u.initial_resources, 1e-6)) for u in uavs]
    load_balance_score = 1 / (1 + np.var(load_ratios)) if len(load_ratios) > 1 else 1.0

    total_consumed = np.sum(list(uav_expenditure.values()), axis=0)
    total_initial_supply = np.sum([u.initial_resources for u in uavs], axis=0)
    resource_utilization_rate = np.mean(total_consumed / np.maximum(total_initial_supply, 1e-6))

    # 归一化各项指标到0-1之间
    # 更新全局最优值
    GLOBAL_BEST_METRICS['completion_rate'] = max(GLOBAL_BEST_METRICS['completion_rate'], completion_rate)
    GLOBAL_BEST_METRICS['satisfied_targets_rate'] = max(GLOBAL_BEST_METRICS['satisfied_targets_rate'], satisfied_targets_rate)
    GLOBAL_BEST_METRICS['sync_feasibility_rate'] = max(GLOBAL_BEST_METRICS['sync_feasibility_rate'], sync_feasibility_rate)
    GLOBAL_BEST_METRICS['load_balance_score'] = max(GLOBAL_BEST_METRICS['load_balance_score'], load_balance_score)
    GLOBAL_BEST_METRICS['resource_utilization_rate'] = max(GLOBAL_BEST_METRICS['resource_utilization_rate'], resource_utilization_rate)
    GLOBAL_BEST_METRICS['total_distance'] = min(GLOBAL_BEST_METRICS['total_distance'], total_distance) if GLOBAL_BEST_METRICS['total_distance'] != float('inf') else total_distance

    # 相对归一化
    norm_completion = completion_rate / GLOBAL_BEST_METRICS['completion_rate'] if GLOBAL_BEST_METRICS['completion_rate'] > 0 else 0
    norm_satisfied = satisfied_targets_rate / GLOBAL_BEST_METRICS['satisfied_targets_rate'] if GLOBAL_BEST_METRICS['satisfied_targets_rate'] > 0 else 0
    norm_sync = sync_feasibility_rate / GLOBAL_BEST_METRICS['sync_feasibility_rate'] if GLOBAL_BEST_METRICS['sync_feasibility_rate'] > 0 else 0
    norm_load = load_balance_score / GLOBAL_BEST_METRICS['load_balance_score'] if GLOBAL_BEST_METRICS['load_balance_score'] > 0 else 0
    norm_utilization = resource_utilization_rate / GLOBAL_BEST_METRICS['resource_utilization_rate'] if GLOBAL_BEST_METRICS['resource_utilization_rate'] > 0 else 0
    norm_distance = 1 - (total_distance / GLOBAL_BEST_METRICS['total_distance']) if GLOBAL_BEST_METRICS['total_distance'] > 0 else 0

    # 计算总评分（归一化后的加权和）
    total_reward_score = (
        norm_completion * 5.0 + 
        norm_satisfied * 3.0 - 
        resource_penalty * 2.0 +
        norm_sync * 1.0 + 
        norm_load * 0.5 + 
        norm_utilization * 0.5 -
        (1 - norm_distance) * 0.1
    ) * 100  # 乘以100使得分数在合理范围内
    
    if is_deadlocked:
        total_reward_score -= 10 * deadlocked_uav_count

    return {
        'total_reward_score': round(total_reward_score, 2), 'is_deadlocked': is_deadlocked,
        'deadlocked_uav_count': deadlocked_uav_count, 'completion_rate': round(completion_rate, 4),
        'satisfied_targets_count': satisfied_targets_count, 
        'total_targets': len(targets),
        'satisfied_targets_rate': round(satisfied_targets_rate, 4),
        'satisfied_targets_rule': '全满足=1 | 全不满足=0 | 部分满足=平方比例',
        'resource_utilization_rate': round(resource_utilization_rate, 4), 'resource_penalty': round(resource_penalty, 4),
        'sync_feasibility_rate': round(sync_feasibility_rate, 4), 'load_balance_score': round(load_balance_score, 4),
        'total_distance': round(total_distance, 2),
        'total_demand': np.array2string(total_demand, formatter={'float_kind': lambda x: "%.0f" % x}),
        'total_contribution': np.array2string(total_contribution, formatter={'float_kind': lambda x: "%.1f" % x})
    }
    return {
        'total_reward_score': round(total_reward_score, 2),
        'completion_rate': round(completion_rate, 4),
        'norm_completion_rate': round(norm_completion, 4),
        'satisfied_targets_count': satisfied_targets_count,
        'total_targets': len(targets),
        'satisfied_targets_rate': round(satisfied_targets_rate, 4),
        'norm_satisfied_targets_rate': round(norm_satisfied, 4),
        'sync_feasibility_rate': round(sync_feasibility_rate, 4),
        'norm_sync_feasibility_rate': round(norm_sync, 4),
        'load_balance_score': round(load_balance_score, 4),
        'norm_load_balance_score': round(norm_load, 4),
        'resource_utilization_rate': round(resource_utilization_rate, 4),
        'norm_resource_utilization_rate': round(norm_utilization, 4),
        'total_distance': round(total_distance, 2),
        'norm_total_distance': round(norm_distance, 4),
        'satisfied_targets_rule': '全满足=1 | 全不满足=0 | 部分满足=平方比例',
        'resource_penalty': round(resource_penalty, 4),
        'is_deadlocked': is_deadlocked,
        'deadlocked_uav_count': deadlocked_uav_count,
        'total_demand': np.array2string(total_demand, formatter={'float_kind': lambda x: "%.0f" % x}),
        'total_contribution': np.array2string(total_contribution, formatter={'float_kind': lambda x: "%.1f" % x})
    }