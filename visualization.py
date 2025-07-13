# -*- coding: utf-8 -*-
# 文件名: visualization.py
# 描述: 封装所有与结果可视化相关的函数。

import matplotlib.pyplot as plt
import numpy as np
import os
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


def plot_results(uavs, targets, obstacles, task_assignments, paths, scenario_name, config, training_time):
    """(已修订) 综合可视化函数，用于绘制最终的无人机任务路径和状态报告"""
    fig, ax = plt.subplots(figsize=(16, 12))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 绘制实体和障碍物
    ax.scatter([u.position[0] for u in uavs], [u.position[1] for u in uavs], c='blue', marker='s', s=100, label='UAV起始点', zorder=5)
    ax.scatter([t.position[0] for t in targets], [t.position[1] for t in targets], c='red', marker='o', s=100, label='目标', zorder=5)
    for u in uavs: ax.text(u.position[0], u.position[1] + 100, f'UAV-{u.id}', ha='center')
    for t in targets: ax.text(t.position[0], t.position[1] + 100, f'T-{t.id}', ha='center')
    for obs in obstacles: obs.draw(ax)

    # 绘制路径和资源消耗
    colors = plt.cm.jet(np.linspace(0, 1, len(uavs)))
    total_deadlocks = 0
    for i, uav in enumerate(uavs):
        color = colors[i]
        path_key = uav.id
        if path_key in paths and paths[path_key]:
            for segment in paths[path_key]:
                path_points, path_len, is_complete, from_id, to_id = segment
                line_style = '-' if is_complete else ':'
                line_width = 2.0 if is_complete else 1.0
                if not is_complete: total_deadlocks += 1
                ax.plot(path_points[:, 0], path_points[:, 1], linestyle=line_style, color=color, linewidth=line_width, zorder=3, alpha=0.8)
                # 在路径中点标注消耗
                if len(path_points) > 1:
                    mid_idx = len(path_points) // 2
                    mid_point = path_points[mid_idx]
                    # 查找目标以获取资源需求
                    target_obj = next((t for t in targets if t.id == to_id), None)
                    if target_obj:
                        res_text = f"-R{target_obj.resources[0]}"
                        ax.text(mid_point[0], mid_point[1], res_text, color=color, fontsize=9, ha='center', va='bottom',
                                bbox=dict(boxstyle="round,pad=0.2", fc="yellow", ec="black", lw=0.5, alpha=0.7))
    # 构建报告
    report_lines = [
        f"场景: {scenario_name}",
        f"总训练耗时: {training_time:.2f} 秒" if training_time > 0 else "(未进行训练)",
        "-" * 20,
        f"资源满足率: {sum(1 for t in targets if not np.any(t.remaining_resources)) / len(targets):.1%}",
        f"目标完成率: {len(set(t for uav in uavs for t in uav.task_sequence)) / len(targets):.1%}",
        f"检测到的死锁/路径失败: {total_deadlocks}",
        "-" * 20, "任务分配详情:"
    ]
    for uav in uavs:
        sequence_str = ' -> '.join(map(str, uav.task_sequence)) if uav.task_sequence else '无任务'
        report_lines.append(f"  UAV-{uav.id}: {sequence_str}")

    ax.text(1.02, 0.98, '\n'.join(report_lines), transform=ax.transAxes, fontsize=10, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_title(f'无人机协同任务规划结果 - {scenario_name}', fontsize=16)
    ax.set_xlabel('X坐标 (米)')
    ax.set_ylabel('Y坐标 (米)')
    ax.legend(loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    # (已修订) 保存图像，文件名包含参数
    output_folder = "output/visualizations"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    param_str = f"lr{config.LEARNING_RATE}_gamma{config.DISCOUNT_FACTOR}_phi{config.NUM_DIRECTIONS}_phrrt{config.USE_PHRRT}"
    clean_scenario_name = scenario_name.replace(' ', '_').replace(':', '')
    save_path = os.path.join(output_folder, f"{clean_scenario_name}_{param_str}.png")
    
    plt.savefig(save_path, dpi=300)
    print(f"可视化结果已保存至: {save_path}")

    if config.SHOW_VISUALIZATION:
        plt.show()
    else:
        plt.close(fig)