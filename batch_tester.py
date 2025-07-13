# -*- coding: utf-8 -*-
# 文件名: batch_tester.py
# 描述: (最终版) 自动执行强化学习、遗传算法、贪婪算法的批量对比测试。

import os
import pickle
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# --- 核心模块导入 ---
from config import Config
from path_planning import DirectedGraph
from path_planning import calculate_economic_sync_speeds
from visualization import visualize_task_assignments, set_chinese_font
from entities import UAV, Target
from path_planning import Obstacle, CircularObstacle, PolygonalObstacle
from evaluate import evaluate_plan

# --- 求解器模块导入 ---
try:
    # 官方版求解器
    from GASolver import GASolver
    from GreedySolver import GreedySolver
    from main import GraphRLSolver
    # 自包含的对比算法
    from GreedySolver_SelfContained import GreedySolver_SelfContained
    from GASolver_SelfContained import GASolver_SelfContained
except ImportError as e:
    print(f"错误: 无法导入所有必需的求解器。请确保所有求解器文件均在同一目录下。")
    print(f"具体错误: {e}")
    exit()

# 全局字体设置
set_chinese_font(manual_font_path='C:/Windows/Fonts/simhei.ttf')

from evaluate import evaluate_plan



def run_single_test(solver_name, config, base_uavs, base_targets, obstacles, scenario_name_for_run) -> dict:
    """统一的测试执行函数，支持所有求解器，并为每种算法的运行都生成可视化报告。"""
    uavs = [UAV(u.id, u.position, u.heading, u.resources, u.max_distance, u.velocity_range, u.economic_speed) for u in base_uavs]
    targets = [Target(t.id, t.position, t.resources, t.value) for t in base_targets]
    scenario_info = {'num_uavs': len(uavs), 'num_targets': len(targets), 'num_obstacles': len(obstacles)}
    
    plan, deadlocked_tasks = {}, {}
    training_time, planning_time = 0.0, 0.0

    # 确保创建的图是“障碍物感知”的，这依赖于 main.py 已被正确修改
    graph = DirectedGraph(uavs, targets, n_phi=config.GRAPH_N_PHI, obstacles=obstacles)

    if solver_name == "RL":
        from main import run_scenario
        # 对RL算法，run_scenario是一个完整的黑盒流程，其返回的training_time代表总耗时
        plan, training_time, deadlocked_tasks = run_scenario(config, uavs, targets, obstacles, scenario_name_for_run, save_visualization=True, show_visualization=False, save_report=True)
        planning_time = 0.0 # RL的规划与训练统一在run_scenario中，不单独计时
    
    else:
        # 对于非RL算法，我们先获取方案，再统一进行可视化和评估
        solve_start_time = time.time()
        
        if solver_name == "GA":
            solver = GASolver(uavs, targets, obstacles, config)
            plan, training_time, planning_time = solver.solve()
            deadlocked_tasks = {} # 原版GA通过协同规划函数避免死锁
        
        elif solver_name == "Greedy":
            solver = GreedySolver(uavs, targets, graph, config)
            task_assignments, training_time = solver.solve()
            # 原版Greedy依赖外部的协同规划函数
            plan, deadlocked_tasks = calculate_economic_sync_speeds(task_assignments, uavs, targets, graph, obstacles, config)
            planning_time = (time.time() - solve_start_time) - training_time

        elif solver_name == "Greedy_SC":
            solver = GreedySolver_SelfContained(uavs, targets, obstacles, graph, config)
            plan, training_time, planning_time = solver.solve()
            deadlocked_tasks = {} # 自包含算法的失败体现在评估指标中
            
        elif solver_name == "GA_SC":
            solver = GASolver_SelfContained(uavs, targets, obstacles, config)
            plan, training_time, planning_time = solver.solve()
            deadlocked_tasks = {} # 自包含算法的失败体现在评估指标中

        else:
            raise ValueError(f"未知的求解器名称: {solver_name}")
        
        # 为所有非RL算法统一生成可视化报告
        visualize_task_assignments(plan, base_uavs, base_targets, obstacles, config, scenario_name_for_run, training_time, planning_time, save_report=True, deadlocked_tasks=deadlocked_tasks)

    # 统一进行结果评估
    quality_metrics = evaluate_plan(plan, base_uavs, base_targets, deadlocked_tasks)
    performance_metrics = {'training_time': round(training_time, 2), 'planning_time': round(planning_time, 2), 'total_time': round(training_time + planning_time, 2)}
    
    return {**scenario_info, **performance_metrics, **quality_metrics}


def batch_test(scenarios_base_dir='scenarios', results_filename='batch_test_results.csv', modes_to_test=None, solvers_to_run=None):
    """主批量测试函数，循环场景和求解器，执行测试并记录结果。"""
    print("开始批量测试...")
    
    # 定义所有可用的求解器
    ALL_SOLVERS = {
        "RL": GraphRLSolver, 
        "GA": GASolver, 
        "Greedy": GreedySolver,
        "Greedy_SC": GreedySolver_SelfContained,
        "GA_SC": GASolver_SelfContained
    }
    
    if solvers_to_run:
        solvers_to_test = {name: cls for name, cls in ALL_SOLVERS.items() if name in solvers_to_run}
        if not solvers_to_test:
            print(f"错误: 指定的求解器 {solvers_to_run} 均无效。可用选项: {list(ALL_SOLVERS.keys())}")
            return
    else:
        solvers_to_test = ALL_SOLVERS
        
    print(f"将要执行的求解器: {list(solvers_to_test.keys())}")
    
    # 断点续传逻辑
    completed_tests = set()
    if os.path.exists(results_filename) and os.path.getsize(results_filename) > 0:
        print(f"检测到已存在的结果文件: {results_filename}，将从中继点继续。")
        try:
            results_df = pd.read_csv(results_filename)
            required_cols = {'scenario', 'solver', 'config', 'obstacle_mode'}
            if required_cols.issubset(results_df.columns):
                for _, row in results_df.iterrows():
                    completed_tests.add((row['scenario'], row['solver'], row['config'], row['obstacle_mode']))
        except Exception as e:
            print(f"警告: 读取结果文件失败 ({e})，将重新开始测试。")
            
    default_config = Config()
    rl_configs = {"RL_Fast": type('RL_Fast_Config', (Config,), {'USE_PHRRT_DURING_TRAINING': False})(), "RL_Precise": type('RL_Precise_Config', (Config,), {'USE_PHRRT_DURING_TRAINING': True})()}
    
    # 搜寻所有场景文件
    target_dirs = modes_to_test or [d for d in os.listdir(scenarios_base_dir) if os.path.isdir(os.path.join(scenarios_base_dir, d))]
    scenario_files = [os.path.join(scenarios_base_dir, mode_dir, f) for mode_dir in target_dirs if os.path.isdir(os.path.join(scenarios_base_dir, mode_dir)) for f in os.listdir(os.path.join(scenarios_base_dir, mode_dir)) if f.endswith('.pkl')]
    
    if not scenario_files:
        print(f"错误: 未能在目录 '{scenarios_base_dir}' 的模式 {target_dirs} 下找到场景文件。请先运行 data_generator.py。")
        return
        
    # 主测试循环
    for scenario_path in tqdm(sorted(scenario_files), desc="批量测试场景"):
        with open(scenario_path, 'rb') as f: scenario_data = pickle.load(f)
        base_uavs, base_targets, obstacles_data = scenario_data['uavs'], scenario_data['targets'], scenario_data['obstacles']
        scenario_name = scenario_data.get('scenario_name', os.path.basename(scenario_path).replace('.pkl', ''))
        
        for solver_name, solver_class in solvers_to_test.items():
            configs_for_solver = rl_configs if solver_name == "RL" else {f"{solver_name}_Default": default_config}
            for config_name, config in configs_for_solver.items():
                for obs_mode in ['present', 'none']:
                    if obs_mode == 'present' and not obstacles_data: continue
                    test_key = (scenario_name, solver_name, config_name, obs_mode)
                    if test_key in completed_tests:
                        tqdm.write(f"已跳过: {test_key}")
                        continue
                    
                    tqdm.write(f"\n> 正在运行: {scenario_name} | {solver_name} | {config_name} | 障碍: {obs_mode}")
                    current_obstacles = obstacles_data if obs_mode == 'present' else []
                    run_name = f"{scenario_name}_{solver_name}_{config_name}_{obs_mode}_obs"
                    result = run_single_test(solver_name, config, base_uavs, base_targets, current_obstacles, run_name)
                    result.update({'scenario': scenario_name, 'solver': solver_name, 'config': config_name, 'obstacle_mode': obs_mode})
                    is_file_new = not os.path.exists(results_filename) or os.path.getsize(results_filename) == 0
                    pd.DataFrame([result]).to_csv(results_filename, mode='a', header=is_file_new, index=False, encoding='utf-8-sig')
                    completed_tests.add(test_key)
                    
    print(f"\n批量测试完成！所有结果已保存至 '{results_filename}'。")


if __name__ == '__main__':
    print("即将开始批量测试...")
    print("将为您生成图片(./output/images)、文本报告(./output/reports)和CSV结果(./batch_test_results.csv)。")
    print("-" * 30)

    # --- 测试指令示例 ---
    # batch_test(modes_to_test=['specified'])

    # 示例1: 运行所有求解器，测试所有场景 (默认执行)
    batch_test()
    
    # 示例2: 只测试新的自包含算法，用于对比分析
    # batch_test(solvers_to_run=['Greedy_SC', 'GA_SC'])

    # 示例3: 在“资源紧缺”场景下，对比所有算法
    # batch_test(modes_to_test=['resource_starvation'])
    
    # 示例4: 对比官方版Greedy和自包含版Greedy
    # batch_test(modes_to_test=['collaborative', 'mixed'], solvers_to_run=['Greedy', 'Greedy_SC'])