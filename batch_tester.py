# -*- coding: utf-8 -*-
# 文件名: batch_tester.py
# 描述: 增强版批量测试系统 - 支持多算法对比实验和详细性能分析

import os
import pickle
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

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
set_chinese_font()

class BatchTester:
    """增强版批量测试器"""
    
    def __init__(self):
        self.results = []
        self.test_configs = {
            'RL': {
                'RL_Fast': {'USE_PHRRT_DURING_TRAINING': False, 'EPISODES': 500},
                'RL_Precise': {'USE_PHRRT_DURING_TRAINING': True, 'EPISODES': 500},
                'RL_Adaptive': {'USE_ADAPTIVE_TRAINING': True, 'EPISODES': 1000}
            },
            'GA': {
                'GA_Default': {},
                'GA_Large': {'POPULATION_SIZE': 100, 'GENERATIONS': 50},
                'GA_Fast': {'POPULATION_SIZE': 50, 'GENERATIONS': 30}
            },
            'Greedy': {
                'Greedy_Default': {},
                'Greedy_Optimized': {'OPTIMIZATION_LEVEL': 'high'}
            }
        }
        
        # 定义所有可用的求解器
        self.ALL_SOLVERS = {
            "RL": GraphRLSolver, 
            "GA": GASolver, 
            "Greedy": GreedySolver,
            "Greedy_SC": GreedySolver_SelfContained,
            "GA_SC": GASolver_SelfContained
        }
    
    def create_config(self, base_config, config_params):
        """创建自定义配置"""
        config_class = type('CustomConfig', (Config,), config_params)
        return config_class()
    
    def run_single_test(self, solver_name, config, base_uavs, base_targets, obstacles, scenario_name_for_run) -> dict:
        """统一的测试执行函数，支持所有求解器"""
        uavs = [UAV(u.id, u.position, u.heading, u.resources, u.max_distance, u.velocity_range, u.economic_speed) for u in base_uavs]
        targets = [Target(t.id, t.position, t.resources, t.value) for t in base_targets]
        scenario_info = {
            'num_uavs': len(uavs), 
            'num_targets': len(targets), 
            'num_obstacles': len(obstacles),
            'scenario_name': scenario_name_for_run,
            'solver': solver_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        plan, deadlocked_tasks = {}, {}
        training_time, planning_time = 0.0, 0.0
        success = True
        error_message = ""

        try:
            # 确保创建的图是"障碍物感知"的
            graph = DirectedGraph(uavs, targets, n_phi=config.GRAPH_N_PHI, obstacles=obstacles)

            if solver_name == "RL":
                from main import run_scenario
                # 对RL算法，run_scenario是一个完整的黑盒流程
                plan, training_time, deadlocked_tasks = run_scenario(
                    config, uavs, targets, obstacles, scenario_name_for_run, 
                    save_visualization=True, show_visualization=False, save_report=True
                )
                planning_time = 0.0
            
            else:
                # 对于非RL算法，我们先获取方案，再统一进行可视化和评估
                solve_start_time = time.time()
                
                if solver_name == "GA":
                    solver = GASolver(uavs, targets, obstacles, config)
                    plan, training_time, planning_time = solver.solve()
                    deadlocked_tasks = {}
                
                elif solver_name == "Greedy":
                    solver = GreedySolver(uavs, targets, graph, config)
                    task_assignments, training_time = solver.solve()
                    plan, deadlocked_tasks = calculate_economic_sync_speeds(
                        task_assignments, uavs, targets, graph, obstacles, config
                    )
                    planning_time = (time.time() - solve_start_time) - training_time

                elif solver_name == "Greedy_SC":
                    solver = GreedySolver_SelfContained(uavs, targets, obstacles, graph, config)
                    plan, training_time, planning_time = solver.solve()
                    deadlocked_tasks = {}
                    
                elif solver_name == "GA_SC":
                    solver = GASolver_SelfContained(uavs, targets, obstacles, config)
                    plan, training_time, planning_time = solver.solve()
                    deadlocked_tasks = {}

                else:
                    raise ValueError(f"未知的求解器名称: {solver_name}")
                
                # 为所有非RL算法统一生成可视化报告
                try:
                    visualize_task_assignments(
                        plan, base_uavs, base_targets, obstacles, config, 
                        scenario_name_for_run, training_time, planning_time, 
                        save_report=True, deadlocked_tasks=deadlocked_tasks
                    )
                except Exception as e:
                    print(f"可视化生成失败: {e}")

        except Exception as e:
            print(f"{solver_name}算法执行失败: {e}")
            success = False
            error_message = str(e)
            plan, training_time, planning_time = {}, 0.0, 0.0
            deadlocked_tasks = {}

        # 统一进行结果评估
        try:
            quality_metrics = evaluate_plan(plan, base_uavs, base_targets, deadlocked_tasks)
        except Exception as e:
            print(f"评估失败: {e}")
            quality_metrics = {
                'total_reward_score': -1000,
                'completion_rate': 0,
                'satisfied_targets_rate': 0,
                'resource_utilization_rate': 0,
                'load_balance_score': 0,
                'total_distance': 0,
                'is_deadlocked': 1,
                'deadlocked_uav_count': len(base_uavs),
                'sync_feasibility_rate': 0,
                'resource_penalty': 1
            }
        
        performance_metrics = {
            'training_time': round(training_time, 2), 
            'planning_time': round(planning_time, 2), 
            'total_time': round(training_time + planning_time, 2),
            'success': success,
            'error_message': error_message
        }
        
        return {**scenario_info, **performance_metrics, **quality_metrics}

    def batch_test(self, scenarios_base_dir='scenarios', results_filename='batch_test_results.csv', 
                   modes_to_test=None, solvers_to_run=None, configs_to_test=None):
        """主批量测试函数"""
        print("开始增强版批量测试...")
        
        # 确定要测试的求解器
        if solvers_to_run:
            solvers_to_test = {name: cls for name, cls in self.ALL_SOLVERS.items() if name in solvers_to_run}
            if not solvers_to_test:
                print(f"错误: 指定的求解器 {solvers_to_run} 均无效。可用选项: {list(self.ALL_SOLVERS.keys())}")
                return
        else:
            solvers_to_test = self.ALL_SOLVERS
        
        print(f"将要执行的求解器: {list(solvers_to_test.keys())}")
        
        # 断点续传逻辑
        completed_tests = set()
        if os.path.exists(results_filename) and os.path.getsize(results_filename) > 0:
            print(f"检测到已存在的结果文件: {results_filename}，将从中继点继续。")
            try:
                results_df = pd.read_csv(results_filename)
                required_cols = {'scenario_name', 'solver', 'config_name', 'obstacle_mode'}
                if required_cols.issubset(results_df.columns):
                    for _, row in results_df.iterrows():
                        completed_tests.add((row['scenario_name'], row['solver'], row['config_name'], row['obstacle_mode']))
            except Exception as e:
                print(f"警告: 读取结果文件失败 ({e})，将重新开始测试。")
        
        # 搜寻所有场景文件
        target_dirs = modes_to_test or [d for d in os.listdir(scenarios_base_dir) if os.path.isdir(os.path.join(scenarios_base_dir, d))]
        scenario_files = []
        for mode_dir in target_dirs:
            mode_path = os.path.join(scenarios_base_dir, mode_dir)
            if os.path.isdir(mode_path):
                for f in os.listdir(mode_path):
                    if f.endswith('.pkl'):
                        scenario_files.append(os.path.join(mode_path, f))
        
        if not scenario_files:
            print(f"错误: 未能在目录 '{scenarios_base_dir}' 的模式 {target_dirs} 下找到场景文件。请先运行 data_generator.py。")
            return
        
        # 主测试循环
        total_tests = 0
        completed_count = 0
        
        for scenario_path in tqdm(sorted(scenario_files), desc="批量测试场景"):
            try:
                with open(scenario_path, 'rb') as f: 
                    scenario_data = pickle.load(f)
                base_uavs, base_targets, obstacles_data = scenario_data['uavs'], scenario_data['targets'], scenario_data['obstacles']
                scenario_name = scenario_data.get('scenario_name', os.path.basename(scenario_path).replace('.pkl', ''))
                
                for solver_name, solver_class in solvers_to_test.items():
                    # 获取该求解器的配置
                    solver_configs = self.test_configs.get(solver_name, {f"{solver_name}_Default": {}})
                    
                    for config_name, config_params in solver_configs.items():
                        # 创建配置
                        if solver_name == "RL":
                            config = self.create_config(Config, config_params)
                        else:
                            config = Config()
                            for key, value in config_params.items():
                                setattr(config, key, value)
                        
                        for obs_mode in ['present', 'none']:
                            if obs_mode == 'present' and not obstacles_data: 
                                continue
                            
                            test_key = (scenario_name, solver_name, config_name, obs_mode)
                            total_tests += 1
                            
                            if test_key in completed_tests:
                                tqdm.write(f"已跳过: {test_key}")
                                completed_count += 1
                                continue
                            
                            tqdm.write(f"执行测试: {test_key}")
                            
                            # 准备障碍物数据
                            obstacles = obstacles_data if obs_mode == 'present' else []
                            
                            # 执行测试
                            result = self.run_single_test(
                                solver_name, config, base_uavs, base_targets, 
                                obstacles, f"{scenario_name}_{solver_name}_{config_name}_{obs_mode}"
                            )
                            
                            # 添加额外信息
                            result.update({
                                'config_name': config_name,
                                'obstacle_mode': obs_mode,
                                'config_params': str(config_params)
                            })
                            
                            self.results.append(result)
                            
                            # 实时保存结果
                            self.save_results(results_filename)
                            
                            completed_count += 1
                            
            except Exception as e:
                print(f"处理场景文件 {scenario_path} 时出错: {e}")
                continue
        
        print(f"\n批量测试完成！")
        print(f"总测试数: {total_tests}")
        print(f"完成测试数: {completed_count}")
        print(f"结果已保存到: {results_filename}")
        
        # 生成分析报告
        self.generate_analysis_report(results_filename)

    def save_results(self, filename):
        """保存结果到CSV文件"""
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(filename, index=False)
            print(f"结果已保存到: {filename}")

    def generate_analysis_report(self, results_filename):
        """生成分析报告"""
        if not os.path.exists(results_filename):
            print("结果文件不存在，无法生成分析报告")
            return
        
        df = pd.read_csv(results_filename)
        
        # 创建输出目录
        output_dir = "output/batch_analysis"
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 算法性能对比
        self.plot_algorithm_comparison(df, output_dir)
        
        # 2. 配置参数影响分析
        self.plot_config_analysis(df, output_dir)
        
        # 3. 场景复杂度分析
        self.plot_scenario_analysis(df, output_dir)
        
        # 4. 生成详细报告
        self.generate_detailed_report(df, output_dir)
        
        print(f"分析报告已生成到: {output_dir}")

    def plot_algorithm_comparison(self, df, output_dir):
        """绘制算法性能对比图"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('算法性能对比分析', fontsize=16)
        
        # 筛选成功的测试
        success_df = df[df['success'] == True]
        
        if success_df.empty:
            print("没有成功的测试数据")
            return
        
        # 1. 综合评分对比
        ax1 = axes[0, 0]
        solver_scores = success_df.groupby('solver')['total_reward_score'].agg(['mean', 'std']).sort_values('mean', ascending=False)
        ax1.bar(solver_scores.index, solver_scores['mean'], yerr=solver_scores['std'], capsize=5)
        ax1.set_title('算法综合评分对比')
        ax1.set_ylabel('综合评分')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. 目标满足率对比
        ax2 = axes[0, 1]
        solver_satisfaction = success_df.groupby('solver')['satisfied_targets_rate'].mean().sort_values(ascending=False)
        ax2.bar(solver_satisfaction.index, solver_satisfaction.values)
        ax2.set_title('目标满足率对比')
        ax2.set_ylabel('满足率')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. 资源利用率对比
        ax3 = axes[0, 2]
        solver_utilization = success_df.groupby('solver')['resource_utilization_rate'].mean().sort_values(ascending=False)
        ax3.bar(solver_utilization.index, solver_utilization.values)
        ax3.set_title('资源利用率对比')
        ax3.set_ylabel('利用率')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. 执行时间对比
        ax4 = axes[1, 0]
        solver_time = success_df.groupby('solver')['total_time'].mean().sort_values(ascending=True)
        ax4.bar(solver_time.index, solver_time.values)
        ax4.set_title('执行时间对比')
        ax4.set_ylabel('时间(秒)')
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. 负载均衡性对比
        ax5 = axes[1, 1]
        solver_balance = success_df.groupby('solver')['load_balance_score'].mean().sort_values(ascending=False)
        ax5.bar(solver_balance.index, solver_balance.values)
        ax5.set_title('负载均衡性对比')
        ax5.set_ylabel('均衡性评分')
        ax5.tick_params(axis='x', rotation=45)
        
        # 6. 成功率对比
        ax6 = axes[1, 2]
        solver_success = df.groupby('solver')['success'].mean().sort_values(ascending=False)
        ax6.bar(solver_success.index, solver_success.values)
        ax6.set_title('算法成功率对比')
        ax6.set_ylabel('成功率')
        ax6.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'algorithm_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()

    def plot_config_analysis(self, df, output_dir):
        """绘制配置参数影响分析"""
        # 筛选RL算法的数据
        rl_df = df[df['solver'] == 'RL'].copy()
        
        if rl_df.empty:
            print("没有RL算法的数据")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('RL算法配置参数影响分析', fontsize=16)
        
        # 1. 配置对评分的影响
        ax1 = axes[0, 0]
        config_scores = rl_df.groupby('config_name')['total_reward_score'].agg(['mean', 'std'])
        ax1.bar(config_scores.index, config_scores['mean'], yerr=config_scores['std'], capsize=5)
        ax1.set_title('配置对综合评分的影响')
        ax1.set_ylabel('综合评分')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. 配置对时间的影响
        ax2 = axes[0, 1]
        config_time = rl_df.groupby('config_name')['total_time'].mean()
        ax2.bar(config_time.index, config_time.values)
        ax2.set_title('配置对执行时间的影响')
        ax2.set_ylabel('时间(秒)')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. 障碍物模式影响
        ax3 = axes[1, 0]
        obstacle_impact = rl_df.groupby('obstacle_mode')['total_reward_score'].mean()
        ax3.bar(obstacle_impact.index, obstacle_impact.values)
        ax3.set_title('障碍物模式对评分的影响')
        ax3.set_ylabel('综合评分')
        
        # 4. 场景规模影响
        ax4 = axes[1, 1]
        ax4.scatter(rl_df['num_uavs'] + rl_df['num_targets'], rl_df['total_reward_score'], alpha=0.6)
        ax4.set_xlabel('场景规模(UAV数+目标数)')
        ax4.set_ylabel('综合评分')
        ax4.set_title('场景规模对评分的影响')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'config_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()

    def plot_scenario_analysis(self, df, output_dir):
        """绘制场景复杂度分析"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('场景复杂度分析', fontsize=16)
        
        # 1. 场景规模分布
        ax1 = axes[0, 0]
        df['scenario_size'] = df['num_uavs'] + df['num_targets']
        ax1.hist(df['scenario_size'], bins=20, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('场景规模')
        ax1.set_ylabel('测试次数')
        ax1.set_title('场景规模分布')
        ax1.grid(True)
        
        # 2. 成功率与场景规模关系
        ax2 = axes[0, 1]
        size_success = df.groupby('scenario_size')['success'].mean()
        ax2.plot(size_success.index, size_success.values, 'o-')
        ax2.set_xlabel('场景规模')
        ax2.set_ylabel('成功率')
        ax2.set_title('成功率与场景规模关系')
        ax2.grid(True)
        
        # 3. 各算法在不同规模场景下的表现
        ax3 = axes[1, 0]
        for solver in df['solver'].unique():
            solver_data = df[df['solver'] == solver]
            ax3.scatter(solver_data['scenario_size'], solver_data['total_reward_score'], 
                       alpha=0.6, label=solver)
        ax3.set_xlabel('场景规模')
        ax3.set_ylabel('综合评分')
        ax3.set_title('各算法在不同规模场景下的表现')
        ax3.legend()
        ax3.grid(True)
        
        # 4. 执行时间与场景规模关系
        ax4 = axes[1, 1]
        for solver in df['solver'].unique():
            solver_data = df[df['solver'] == solver]
            ax4.scatter(solver_data['scenario_size'], solver_data['total_time'], 
                       alpha=0.6, label=solver)
        ax4.set_xlabel('场景规模')
        ax4.set_ylabel('执行时间(秒)')
        ax4.set_title('执行时间与场景规模关系')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'scenario_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()

    def generate_detailed_report(self, df, output_dir):
        """生成详细的分析报告"""
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("批量测试详细分析报告")
        report_lines.append("="*80)
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 1. 测试概览
        report_lines.append("1. 测试概览")
        report_lines.append("-"*40)
        report_lines.append(f"总测试数: {len(df)}")
        report_lines.append(f"成功测试数: {len(df[df['success'] == True])}")
        report_lines.append(f"失败测试数: {len(df[df['success'] == False])}")
        report_lines.append(f"总体成功率: {len(df[df['success'] == True])/len(df)*100:.2f}%")
        report_lines.append("")
        
        # 2. 算法性能排名
        report_lines.append("2. 算法性能排名")
        report_lines.append("-"*40)
        success_df = df[df['success'] == True]
        if not success_df.empty:
            solver_ranking = success_df.groupby('solver').agg({
                'total_reward_score': ['mean', 'std'],
                'satisfied_targets_rate': 'mean',
                'resource_utilization_rate': 'mean',
                'total_time': 'mean',
                'load_balance_score': 'mean'
            }).round(4)
            
            for solver in solver_ranking.index:
                report_lines.append(f"\n{solver}算法:")
                report_lines.append(f"  - 平均综合评分: {solver_ranking.loc[solver, ('total_reward_score', 'mean')]:.2f} ± {solver_ranking.loc[solver, ('total_reward_score', 'std')]:.2f}")
                report_lines.append(f"  - 平均目标满足率: {solver_ranking.loc[solver, ('satisfied_targets_rate', 'mean')]:.4f}")
                report_lines.append(f"  - 平均资源利用率: {solver_ranking.loc[solver, ('resource_utilization_rate', 'mean')]:.4f}")
                report_lines.append(f"  - 平均执行时间: {solver_ranking.loc[solver, ('total_time', 'mean')]:.2f}秒")
                report_lines.append(f"  - 平均负载均衡性: {solver_ranking.loc[solver, ('load_balance_score', 'mean')]:.4f}")
        
        # 3. 配置参数影响
        report_lines.append("\n3. 配置参数影响分析")
        report_lines.append("-"*40)
        rl_df = df[df['solver'] == 'RL']
        if not rl_df.empty:
            config_analysis = rl_df.groupby('config_name').agg({
                'total_reward_score': ['mean', 'std'],
                'total_time': 'mean',
                'satisfied_targets_rate': 'mean'
            }).round(4)
            
            for config in config_analysis.index:
                report_lines.append(f"\n{config}配置:")
                report_lines.append(f"  - 平均综合评分: {config_analysis.loc[config, ('total_reward_score', 'mean')]:.2f} ± {config_analysis.loc[config, ('total_reward_score', 'std')]:.2f}")
                report_lines.append(f"  - 平均执行时间: {config_analysis.loc[config, ('total_time', 'mean')]:.2f}秒")
                report_lines.append(f"  - 平均目标满足率: {config_analysis.loc[config, ('satisfied_targets_rate', 'mean')]:.4f}")
        
        # 4. 场景复杂度影响
        report_lines.append("\n4. 场景复杂度影响分析")
        report_lines.append("-"*40)
        df['scenario_size'] = df['num_uavs'] + df['num_targets']
        size_analysis = df.groupby('scenario_size').agg({
            'success': 'mean',
            'total_reward_score': 'mean',
            'total_time': 'mean'
        }).round(4)
        
        for size in size_analysis.index:
            report_lines.append(f"\n场景规模 {size} (UAV+目标):")
            report_lines.append(f"  - 成功率: {size_analysis.loc[size, 'success']:.4f}")
            report_lines.append(f"  - 平均综合评分: {size_analysis.loc[size, 'total_reward_score']:.2f}")
            report_lines.append(f"  - 平均执行时间: {size_analysis.loc[size, 'total_time']:.2f}秒")
        
        # 5. 错误分析
        report_lines.append("\n5. 错误分析")
        report_lines.append("-"*40)
        failed_tests = df[df['success'] == False]
        if not failed_tests.empty:
            error_counts = failed_tests['error_message'].value_counts()
            for error, count in error_counts.head(10).items():
                report_lines.append(f"  - {error}: {count}次")
        else:
            report_lines.append("  无失败测试")
        
        # 保存报告
        report_path = os.path.join(output_dir, 'detailed_analysis_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"详细报告已保存到: {report_path}")

def main():
    """主函数"""
    tester = BatchTester()
    
    # 运行批量测试
    tester.batch_test(
        scenarios_base_dir='scenarios',
        results_filename='batch_test_results_enhanced.csv',
        solvers_to_run=['RL', 'GA', 'Greedy', 'Greedy_SC', 'GA_SC'],
        modes_to_test=['simple', 'complex', 'extreme']
    )

if __name__ == "__main__":
    main()