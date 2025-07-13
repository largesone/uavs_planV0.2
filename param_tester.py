import itertools
import time
import numpy as np
from main import Config, run_scenario
from scenarios import get_small_scenario, get_complex_scenario

def run_parameter_testing():
    """自动化测试不同的超参数组合对RL模型性能的影响。"""
    # --- 定义要测试的参数网格 ---
    # 在这里添加或修改你想要测试的参数和值的范围
    param_grid = {
        'LR': [0.001, 0.0005],
        'GAMMA': [0.99, 0.95],
        'GRAPH_N_PHI': [8, 16],
        'USE_PHRRT_DURING_TRAINING': [False, True]
    }

    # 生成所有参数组合
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"将要测试 {len(param_combinations)} 种不同的参数组合...")

    # --- 选择测试场景 ---
    # 你可以根据需要切换场景，例如 get_complex_scenario
    base_uavs, base_targets, base_obstacles = get_small_scenario(obstacle_tolerance=50.0)
    scenario_name_base = "ParamTest_SmallScenario"

    results = []

    for i, params in enumerate(param_combinations):
        print(f"\n{'='*40}\n[测试 {i+1}/{len(param_combinations)}] 参数: {params}\n{'='*40}")
        
        # --- 应用参数 ---
        config = Config()
        config.RUN_TRAINING = True  # 强制进行训练
        for key, value in params.items():
            setattr(config, key, value)
        
        # [更新] 确保场景名称的格式与main.py中的逻辑一致，便于追踪
        # 不再使用hash，直接用可读参数
        param_str = f"lr{params['LR']}_gamma{params['GAMMA']}_phi{params['GRAPH_N_PHI']}_phrrt{params['USE_PHRRT_DURING_TRAINING']}"
        scenario_name = f"{scenario_name_base}_{param_str}"

        # --- 运行场景 ---
        start_time = time.time()
        final_plan, training_time, deadlocked_tasks = run_scenario(
            config=config,
            base_uavs=base_uavs,
            base_targets=base_targets,
            obstacles=base_obstacles,
            scenario_name=scenario_name_base, # 主场景名保持一致，具体参数体现在模型路径中
            save_visualization=False, # 在批量测试时通常关闭可视化以加快速度
            show_visualization=False,
            save_report=False
        )
        end_time = time.time()

        # --- 评估和记录结果 ---
        num_completed_targets = 0
        if final_plan:
            all_assigned_targets = {task['target_id'] for tasks in final_plan.values() for task in tasks}
            num_completed_targets = len(all_assigned_targets)

        total_runtime = end_time - start_time
        num_deadlocks = sum(1 for tasks in deadlocked_tasks.values() if tasks)

        results.append({
            'params': params,
            'training_time': training_time,
            'total_runtime': total_runtime,
            'completed_targets': num_completed_targets,
            'num_deadlocks': num_deadlocks
        })

        print(f"--- 测试 {i+1} 完成 --- 耗时: {total_runtime:.2f}s, 完成目标: {num_completed_targets}, 死锁: {num_deadlocks}")

    # --- 结果排序和展示 ---
    # 按完成的目标数（降序）和训练时间（升序）对结果进行排序
    sorted_results = sorted(results, key=lambda x: (-x['completed_targets'], x['training_time']))

    print(f"\n\n{'='*50}\n========= 参数测试最终排名 =========\n{'='*50}")
    for rank, res in enumerate(sorted_results):
        print(f"\n--- 排名 {rank+1} ---")
        print(f"  参数: {res['params']}")
        print(f"  性能: 完成 {res['completed_targets']} 个目标 | 死锁 {res['num_deadlocks']} 个")
        print(f"  耗时: 训练 {res['training_time']:.2f}s | 总计 {res['total_runtime']:.2f}s")

if __name__ == "__main__":
    run_parameter_testing()