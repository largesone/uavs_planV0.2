# -*- coding: utf-8 -*-
# 文件名: visualization.py
# 描述: 封装所有与结果可视化相关的函数。

import matplotlib.pyplot as plt
import numpy as np
import os
import time
from collections import defaultdict
from path_planning import PHCurveRRTPlanner

# =============================================================================
# section 5: 结果可视化与报告
# =============================================================================

def set_chinese_font():
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False


def visualize_task_assignments(task_assignments, uavs, targets, graph, config):
    """(从main.py迁移) 可视化任务分配方案"""
    fig = plt.figure(figsize=(12, 8))
    set_chinese_font()
    
    # 原main.py中的可视化逻辑...
    plt.title(f'任务分配方案 (φ={config.NUM_DIRECTIONS})')
    plt.savefig('output/task_assignments.png')
    plt.close(fig)


def plot_results(uavs, targets, obstacles, task_assignments, paths, scenario_name, config, training_time, plan_generation_time=None, evaluation_metrics=None):
    """(已优化) 综合可视化函数，实现详细资源跟踪和多维度任务报告"""
    fig, ax = plt.subplots(figsize=(22, 14))
    ax.set_facecolor("#f0f0f0")
    set_chinese_font()

    # 1. 事件驱动的资源跟踪系统
    temp_uav_resources = {u.id: u.initial_resources.copy().astype(float) for u in uavs}
    temp_target_resources = {t.id: t.resources.copy().astype(float) for t in targets}
    events = defaultdict(list)
    target_collaborators_details = defaultdict(list)

    # 2. 准备路径数据和事件分组
    for uav in uavs:
        path_key = uav.id
        if path_key in paths and paths[path_key]:
            for step, segment in enumerate(paths[path_key], 1):
                path_points, path_len, is_complete, from_id, to_id = segment
                if to_id is None: continue
                events[(to_id, step)].append({'uav_id': uav.id, 'step': step, 'path_points': path_points, 'is_complete': is_complete})
                target = next((t for t in targets if t.id == to_id), None)
                if target: target_collaborators_details[to_id].append({'uav_id': uav.id, 'step': step})

    # 3. 绘制实体和障碍物
    ax.scatter([u.position[0] for u in uavs], [u.position[1] for u in uavs], c='blue', marker='s', s=150, label='无人机起点', zorder=5, edgecolors='black')
    ax.scatter([t.position[0] for t in targets], [t.position[1] for t in targets], c='red', marker='o', s=150, label='目标', zorder=5, edgecolors='black')

    # 4. UAV和目标标注
    for u in uavs:
        ax.annotate(f"UAV{u.id}", (u.position[0], u.position[1]), fontsize=12, fontweight='bold',
                    xytext=(0, -25), textcoords='offset points', ha='center', va='top')
        ax.annotate(f"初始: {np.array2string(u.initial_resources, formatter={'float_kind':lambda x: f'{x:.0f}'})}",
                    (u.position[0], u.position[1] + 100), fontsize=8, xytext=(0, 10), textcoords='offset points', ha='center', va='bottom', color='navy')

    for t in targets:
        demand_str = np.array2string(t.resources, formatter={'float_kind':lambda x: f'{x:.0f}'})
        ax.annotate(f"T{t.id}", (t.position[0], t.position[1]), fontsize=12, fontweight='bold',
                    xytext=(0, -25), textcoords='offset points', ha='center', va='top')
        ax.annotate(f"需求: {demand_str}", (t.position[0], t.position[1] + 100), fontsize=8, xytext=(0, 10), textcoords='offset points', ha='center', va='bottom', color='red')
    for obs in obstacles: obs.draw(ax)

    # 5. 绘制路径和资源消耗
    colors = plt.cm.jet(np.linspace(0, 1, len(uavs)))
    uav_color_map = {u.id: colors[i] for i, u in enumerate(uavs)}
    total_deadlocks = 0

    for uav in uavs:
        uav_id = uav.id
        color = uav_color_map.get(uav_id, 'gray')
        path_key = uav_id
        temp_resources = uav.initial_resources.copy().astype(float)

        if path_key in paths and paths[path_key]:
            for step, segment in enumerate(paths[path_key], 1):
                path_points, path_len, is_complete, from_id, to_id = segment
                if not is_complete: total_deadlocks += 1
                line_style = '-' if is_complete else '--'
                line_width = 2 if is_complete else 1.5

                # 绘制路径
                if path_points is not None and len(path_points) > 1:
                    ax.plot(path_points[:, 0], path_points[:, 1], color=color, linestyle=line_style, 
                            linewidth=line_width, alpha=0.9, zorder=3)

                    # 步骤编号标注
                    num_pos = path_points[int(len(path_points) * 0.45)]
                    ax.text(num_pos[0], num_pos[1], str(step), color='white', backgroundcolor=color,
                            ha='center', va='center', fontsize=9, fontweight='bold',
                            bbox=dict(boxstyle='circle,pad=0.2', fc=color, ec='none'), zorder=4)

                    # 资源消耗计算与标注
                    target = next((t for t in targets if t.id == to_id), None)
                    if target:
                        resource_cost = np.minimum(temp_resources, temp_target_resources[target.id])
                        temp_resources -= resource_cost
                        temp_target_resources[target.id] -= resource_cost

                        # 资源剩余标注
                        res_pos = path_points[int(len(path_points) * 0.85)]
                        res_text = f"R: {np.array2string(temp_resources.clip(0), formatter={'float_kind':lambda x: f'{x:.0f}'})}"
                        ax.text(res_pos[0], res_pos[1], res_text, color=color, fontsize=7, fontweight='bold',
                                ha='center', va='center', bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=color, alpha=0.8, lw=0.5), zorder=7)
    # 6. 目标详细标注
    for t in targets:
        target_id = t.id
        demand_str = np.array2string(t.resources, formatter={'float_kind':lambda x: f'{x:.0f}'})
        remaining_str = np.array2string(temp_target_resources[target_id].clip(0), formatter={'float_kind':lambda x: f'{x:.0f}'})
        annotation_text = f"目标 {target_id}\n总需求: {demand_str}\n剩余需求: {remaining_str}\n------------------"
        collaborators = target_collaborators_details.get(target_id, [])
        if collaborators:
            for detail in collaborators:
                uav = next((u for u in uavs if u.id == detail['uav_id']), None)
                if uav: annotation_text += f"\nUAV {uav.id} (步骤{detail['step']})"

        # 状态指示（满足/未满足）
        if np.all(temp_target_resources[target_id] < 1e-6):
            status_str, bbox_color = "[完成] 需求满足", 'lightgreen'
        else:
            status_str, bbox_color = "[进行中] 资源不足", 'mistyrose'
        annotation_text += f"\n------------------\n状态: {status_str}"

        # 绘制标注
        ax.annotate(annotation_text, (t.position[0], t.position[1]), fontsize=7,
                    xytext=(15, -15), textcoords='offset points', ha='left', va='top',
                    bbox=dict(boxstyle='round,pad=0.4', fc=bbox_color, ec='black', alpha=0.9, lw=0.5), zorder=8)

    # 7. 构建综合报告
    deadlock_summary_text = """
!!! 检测到 {total_deadlocks} 处路径失败/死锁 !!!
--------------------------
""" if total_deadlocks > 0 else ""

    # 资源满足率计算
    resource_types = len(targets[0].resources) if targets else 2
    satisfied_targets_count = sum(1 for t in targets if np.all(temp_target_resources[t.id] < 1e-6))
    num_targets = len(targets)
    satisfaction_rate_percent = (satisfied_targets_count / num_targets * 100) if num_targets > 0 else 100

    # 总体资源统计
    total_demand_all = np.sum([t.resources for t in targets], axis=0) if targets else np.zeros(resource_types)
    total_contribution_all = np.sum([u.initial_resources - temp_uav_resources[u.id] for u in uavs], axis=0)
    total_demand_safe = np.where(total_demand_all < 1e-6, 1e-6, total_demand_all)
    overall_completion_rate_percent = np.mean(np.minimum(total_contribution_all, total_demand_all) / total_demand_safe) * 100

    # 报告头部
    report_header = [
        f"---------- {scenario_name} 执行报告 ----------",
        f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"训练耗时: {training_time:.2f}s" if training_time else "训练模式: 未启用",
        f"规划耗时: {plan_generation_time:.2f}s" if plan_generation_time else "规划模式: 快速估算",
        "-" * 50,
        "资源汇总统计:",
        f"- 总需求资源: {np.array2string(total_demand_all, formatter={'float_kind':lambda x: f'{x:.0f}'})}",
        f"- 总贡献资源: {np.array2string(total_contribution_all, formatter={'float_kind':lambda x: f'{x:.1f}'})}",
        f"- 资源完成率: {overall_completion_rate_percent:.1f}%",
        f"- 目标完成率: {satisfied_targets_count}/{num_targets} ({satisfaction_rate_percent:.1f}%)",
        deadlock_summary_text,
    ]

    # 添加评估指标
    if evaluation_metrics:
        report_header.extend(["-" * 50, "性能评估指标:"])
        for key, value in evaluation_metrics.items():
            if isinstance(value, float):
                report_header.append(f"- {key}: {value:.4f}")
            else:
                report_header.append(f"- {key}: {value}")

    # 任务分配详情
    report_body = ["-" * 50, "任务分配详情:"]
    for uav in uavs:
        path_key = uav.id
        path_details = paths.get(path_key, [])
        sequence = []
        for step, segment in enumerate(path_details, 1):
            _, _, is_complete, _, to_id = segment
            status = "✓" if is_complete else "✗"
            sequence.append(f"T{to_id}{status}")
        sequence_str = ' -> '.join(sequence) if sequence else '无任务'
        report_body.append(f"UAV-{uav.id}:")
        report_body.append(f"  任务序列: {sequence_str}")
        report_body.append(f"  初始资源: {np.array2string(uav.initial_resources, formatter={'float_kind':lambda x: f'{x:.0f}'})}")
        report_body.append(f"  剩余资源: {np.array2string(temp_uav_resources[uav.id].clip(0), formatter={'float_kind':lambda x: f'{x:.0f}'})}")
        report_body.append("")  # 空行分隔

    # 组合报告并绘制
    final_report = '\n'.join(report_header + report_body)
    plt.subplots_adjust(right=0.75)
    fig.text(0.77, 0.95, final_report, transform=plt.gcf().transFigure, ha="left", va="top", fontsize=9,
             bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', ec='grey', alpha=0.9))

    # 8. 标题和样式设置
    train_mode_str = '高精度PH-RRT' if config.USE_PHRRT else '快速RRT'
    title_text = (
        f"多无人机协同任务分配与路径规划\n"
        f"场景: {scenario_name} | UAV: {len(uavs)} | 目标: {len(targets)} | 障碍: {len(obstacles)}\n"
        f"算法模式: {train_mode_str} | 资源完成率: {overall_completion_rate_percent:.1f}%"
    )
    ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)

    ax.set_xlabel("X坐标 (米)", fontsize=12)
    ax.set_ylabel("Y坐标 (米)", fontsize=12)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_aspect('equal', adjustable='box')

    # 9. 保存和显示设置
    output_folder = "output/visualizations"
    os.makedirs(output_folder, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    clean_scenario_name = scenario_name.replace(' ', '_').replace(':', '')
    save_path = os.path.join(output_folder, f"{clean_scenario_name}_{timestamp}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[可视化] 结果已保存至: {save_path}")

    # 生成文本报告
    report_dir = "output/reports"
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f"{clean_scenario_name}_{timestamp}.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(final_report)
    print(f"[报告] 详细结果已保存至: {report_path}")

    if config.SHOW_VISUALIZATION:
        plt.show()
    plt.close(fig)