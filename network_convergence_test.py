# -*- coding: utf-8 -*-
# 文件名: network_convergence_test.py
# 描述: 网络收敛性测试编排脚本 - 增强版本，支持TensorBoard和统一目录管理

import os
import time
import pickle
import json
import matplotlib.pyplot as plt
import numpy as np
import shutil
from typing import List, Dict, Any
from matplotlib.font_manager import FontProperties, findfont

# 导入核心模块
from main import run_scenario, set_chinese_font, cleanup_temp_files
from scenarios import get_small_scenario, get_complex_scenario, get_strategic_trap_scenario, get_new_experimental_scenario
from config import Config

class NetworkConvergenceTester:
    """网络收敛性测试器 - 增强版本，支持TensorBoard和详细分析"""
    
    def __init__(self):
        """初始化测试器"""
        self.config = Config()
        self.results = {}
        self.training_histories = {}
        
        # 定义测试配置
        self.network_types = ["SimpleNetwork", "DeepFCN", "GAT", "DeepFCNResidual"]
        self.scenarios = {
            "simple_convergence": get_small_scenario,
            "experimental_scenario": get_new_experimental_scenario,
            "strategic_trap": get_strategic_trap_scenario
        }
        
        # 创建统一的输出目录结构
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        self.output_dir = f"output/convergence_test_{timestamp}"
        self.create_directory_structure()
        
        # 设置中文字体
        set_chinese_font()
        
        print(f"网络收敛性测试系统 - 增强版本")
        print(f"=" * 60)
        print(f"输出目录: {self.output_dir}")
        print(f"测试网络类型: {self.network_types}")
        print(f"测试场景: {list(self.scenarios.keys())}")
        print(f"TensorBoard支持: 启用")
        print(f"详细分析: 启用")
        print(f"=" * 60)
        print(f"详细分析: 启用")
        print(f"=" * 60)
    
    def create_directory_structure(self):
        """创建统一的目录结构"""
        directories = [
            self.output_dir,
            f"{self.output_dir}/experiments",  # 各个实验的详细结果
            f"{self.output_dir}/comparisons",  # 对比分析
            f"{self.output_dir}/tensorboard",  # TensorBoard日志汇总
            f"{self.output_dir}/summary",      # 汇总报告
            f"{self.output_dir}/archive"       # 原始数据归档
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        print(f"已创建目录结构: {self.output_dir}")
    
    def run_all_tests(self):
        """运行所有测试 - 增强版本"""
        print("\n" + "=" * 60)
        print("开始网络收敛性测试")
        print("=" * 60)
        
        total_tests = len(self.network_types) * len(self.scenarios)
        current_test = 0
        
        for network_type in self.network_types:
            for scenario_name, scenario_func in self.scenarios.items():
                current_test += 1
                print(f"\n{'=' * 60}")
                print(f"测试 {current_test}/{total_tests}: {network_type} + {scenario_name}")
                print(f"{'=' * 60}")
                
                test_key = f"{network_type}_{scenario_name}"
                experiment_dir = f"{self.output_dir}/experiments/{test_key}"
                
                try:
                    # 获取场景数据
                    uavs, targets, obstacles = scenario_func(50.0)
                    
                    print(f"场景配置: {len(uavs)} UAVs, {len(targets)} 目标, {len(obstacles)} 障碍物")
                    
                    # 运行场景，使用统一的输出目录
                    final_plan, training_time, training_history, evaluation_metrics = run_scenario(
                        self.config, uavs, targets, obstacles, scenario_name,
                        network_type=network_type,
                        save_visualization=True,
                        show_visualization=False,
                        output_base_dir=f"{self.output_dir}/experiments"
                    )
                    
                    # 保存结果
                    self.results[test_key] = {
                        "network_type": network_type,
                        "scenario_name": scenario_name,
                        "training_time": training_time,
                        "evaluation_metrics": evaluation_metrics,
                        "final_plan": final_plan,
                        "experiment_dir": experiment_dir,
                        "uav_count": len(uavs),
                        "target_count": len(targets),
                        "obstacle_count": len(obstacles)
                    }
                    
                    self.training_histories[test_key] = training_history
                    
                    # 输出测试结果摘要
                    print(f"\n测试完成: {test_key}")
                    print(f"  训练时间: {training_time:.2f}秒")
                    print(f"  实验目录: {experiment_dir}")
                    
                    if evaluation_metrics:
                        print(f"  完成率: {evaluation_metrics.get('completion_rate', 0):.4f}")
                        print(f"  总奖励: {evaluation_metrics.get('total_reward_score', 0):.2f}")
                        print(f"  目标满足率: {evaluation_metrics.get('satisfied_targets_rate', 0):.4f}")
                    
                    # 生成单个测试的收敛性分析
                    self.analyze_single_test_convergence(test_key, training_history)
                
                except Exception as e:
                    print(f"测试失败: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    
                    self.results[test_key] = {
                        "error": str(e),
                        "network_type": network_type,
                        "scenario_name": scenario_name,
                        "experiment_dir": experiment_dir
                    }
        
        # 生成综合分析报告
        print(f"\n{'=' * 60}")
        print("生成综合分析报告...")
        print(f"{'=' * 60}")
        
        self.generate_summary_report()
        self.generate_comparison_plots()
        self.generate_convergence_recommendations()
        self.create_tensorboard_summary()
        
        print(f"\n{'=' * 60}")
        print("网络收敛性测试完成!")
        print(f"结果保存在: {self.output_dir}")
        print(f"{'=' * 60}")
    
    def analyze_single_test_convergence(self, test_key, training_history):
        """分析单个测试的收敛性"""
        if not training_history.get('episode_rewards'):
            return
        
        rewards = training_history['episode_rewards']
        
        # 简单的收敛性判断
        convergence_status = "未收敛"
        if len(rewards) > 100:
            recent_std = np.std(rewards[-50:])
            overall_std = np.std(rewards)
            if recent_std < overall_std * 0.3:
                convergence_status = "已收敛"
            elif recent_std < overall_std * 0.6:
                convergence_status = "部分收敛"
        
        print(f"  收敛状态: {convergence_status}")
        
        # 保存到结果中
        if test_key in self.results:
            self.results[test_key]['convergence_status'] = convergence_status
    
    def generate_convergence_recommendations(self):
        """生成收敛性改进建议"""
        recommendations_path = f"{self.output_dir}/summary/convergence_recommendations.txt"
        
        with open(recommendations_path, 'w', encoding='utf-8') as f:
            f.write("网络收敛性改进建议\n")
            f.write("=" * 50 + "\n\n")
            
            # 分析各网络类型的表现
            network_performance = {}
            for test_key, result in self.results.items():
                if 'error' not in result:
                    network_type = result['network_type']
                    if network_type not in network_performance:
                        network_performance[network_type] = []
                    
                    metrics = result.get('evaluation_metrics', {})
                    completion_rate = metrics.get('completion_rate', 0)
                    total_reward = metrics.get('total_reward_score', 0)
                    
                    network_performance[network_type].append({
                        'completion_rate': completion_rate,
                        'total_reward': total_reward,
                        'training_time': result['training_time'],
                        'convergence_status': result.get('convergence_status', '未知')
                    })
            
            # 生成建议
            f.write("1. 网络类型分析:\n")
            for network_type, performances in network_performance.items():
                if performances:
                    avg_completion = np.mean([p['completion_rate'] for p in performances])
                    avg_reward = np.mean([p['total_reward'] for p in performances])
                    avg_time = np.mean([p['training_time'] for p in performances])
                    
                    f.write(f"\n{network_type}:\n")
                    f.write(f"  - 平均完成率: {avg_completion:.3f}\n")
                    f.write(f"  - 平均奖励: {avg_reward:.2f}\n")
                    f.write(f"  - 平均训练时间: {avg_time:.2f}秒\n")
                    
                    # 生成具体建议
                    if avg_completion < 0.3:
                        f.write(f"  - 建议: 完成率较低，考虑增加网络深度或调整奖励函数\n")
                    elif avg_completion > 0.8:
                        f.write(f"  - 建议: 表现良好，可作为基准网络\n")
                    else:
                        f.write(f"  - 建议: 表现中等，可尝试调整超参数\n")
            
            f.write(f"\n2. 通用改进建议:\n")
            f.write(f"  - 如果所有网络收敛都较差，建议:\n")
            f.write(f"    * 降低学习率 (当前: {self.config.LEARNING_RATE})\n")
            f.write(f"    * 增加训练回合数 (当前: {self.config.training_config.episodes})\n")
            f.write(f"    * 调整奖励函数权重\n")
            f.write(f"    * 检查环境设计是否合理\n")
            f.write(f"  - 如果训练不稳定，建议:\n")
            f.write(f"    * 增加经验回放池大小\n")
            f.write(f"    * 降低探索率衰减速度\n")
            f.write(f"    * 使用梯度裁剪\n")
            
            f.write(f"\n3. 下一步实验建议:\n")
            f.write(f"  - 实施Double DQN减少过估计\n")
            f.write(f"  - 尝试优先经验回放\n")
            f.write(f"  - 考虑使用Dueling DQN\n")
            f.write(f"  - 实验不同的网络初始化方法\n")
        
        print(f"收敛性改进建议已保存至: {recommendations_path}")
    
    def create_tensorboard_summary(self):
        """创建TensorBoard汇总"""
        summary_dir = f"{self.output_dir}/tensorboard"
        
        # 创建汇总脚本
        script_path = f"{self.output_dir}/view_tensorboard.bat"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write("@echo off\n")
            f.write("echo 启动TensorBoard查看训练结果...\n")
            f.write(f"tensorboard --logdir={summary_dir}\n")
            f.write("pause\n")
        
        # 创建说明文件
        readme_path = f"{self.output_dir}/tensorboard/README.txt"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write("TensorBoard使用说明\n")
            f.write("=" * 30 + "\n\n")
            f.write("1. 安装TensorBoard: pip install tensorboard\n")
            f.write(f"2. 运行命令: tensorboard --logdir={summary_dir}\n")
            f.write("3. 在浏览器中打开: http://localhost:6006\n\n")
            f.write("各实验的TensorBoard日志位于对应的实验目录下的tensorboard文件夹中\n")
        
        print(f"TensorBoard汇总已创建: {summary_dir}")
        print(f"使用脚本启动: {script_path}")

    def generate_summary_report(self):
        """生成汇总报告 - 增强版本"""
        report_path = f"{self.output_dir}/summary/summary_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("网络收敛性测试汇总报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 统计信息
            total_tests = len(self.results)
            successful_tests = sum(1 for r in self.results.values() if 'error' not in r)
            failed_tests = total_tests - successful_tests
            
            f.write(f"测试统计:\n")
            f.write(f"  总测试数: {total_tests}\n")
            f.write(f"  成功测试: {successful_tests}\n")
            f.write(f"  失败测试: {failed_tests}\n\n")
            
            # 详细结果
            f.write("详细结果:\n")
            f.write("-" * 30 + "\n")
            
            for test_key, result in self.results.items():
                f.write(f"\n测试: {test_key}\n")
                
                if 'error' in result:
                    f.write(f"  状态: 失败\n")
                    f.write(f"  错误: {result['error']}\n")
                else:
                    f.write(f"  状态: 成功\n")
                    f.write(f"  训练时间: {result['training_time']:.2f}秒\n")
                    
                    metrics = result.get('evaluation_metrics', {})
                    if metrics:
                        f.write(f"  完成率: {metrics.get('completion_rate', 0):.4f}\n")
                        f.write(f"  总奖励: {metrics.get('total_reward_score', 0):.2f}\n")
                        f.write(f"  目标满足率: {metrics.get('satisfied_targets_rate', 0):.4f}\n")
            
            # 性能对比
            f.write(f"\n性能对比:\n")
            f.write("-" * 30 + "\n")
            
            successful_results = {k: v for k, v in self.results.items() if 'error' not in v}
            
            if successful_results:
                # 训练时间对比
                training_times = {k: v['training_time'] for k, v in successful_results.items()}
                fastest_test = min(training_times.items(), key=lambda x: x[1])
                f.write(f"最快训练: {fastest_test[0]} ({fastest_test[1]:.2f}秒)\n")
                
                # 完成率对比
                completion_rates = {k: v['evaluation_metrics'].get('completion_rate', 0) 
                                  for k, v in successful_results.items()}
                best_completion = max(completion_rates.items(), key=lambda x: x[1])
                f.write(f"最高完成率: {best_completion[0]} ({best_completion[1]:.4f})\n")
                
                # 总奖励对比
                total_rewards = {k: v['evaluation_metrics'].get('total_reward_score', 0) 
                               for k, v in successful_results.items()}
                best_reward = max(total_rewards.items(), key=lambda x: x[1])
                f.write(f"最高奖励: {best_reward[0]} ({best_reward[1]:.2f})\n")
        
        print(f"汇总报告已保存至: {report_path}")
    
    def generate_comparison_plots(self):
        """生成对比图表"""
        successful_results = {k: v for k, v in self.results.items() if 'error' not in v}
        
        if not successful_results:
            print("没有成功的测试结果，跳过图表生成")
            return
        
        # 创建对比图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('网络收敛性测试对比', fontsize=16)
        
        # 训练时间对比
        test_names = list(successful_results.keys())
        training_times = [successful_results[k]['training_time'] for k in test_names]
        
        axes[0, 0].bar(range(len(test_names)), training_times)
        axes[0, 0].set_title('训练时间对比')
        axes[0, 0].set_ylabel('时间 (秒)')
        axes[0, 0].set_xticks(range(len(test_names)))
        axes[0, 0].set_xticklabels(test_names, rotation=45, ha='right')
        
        # 完成率对比
        completion_rates = [successful_results[k]['evaluation_metrics'].get('completion_rate', 0) 
                           for k in test_names]
        
        axes[0, 1].bar(range(len(test_names)), completion_rates)
        axes[0, 1].set_title('完成率对比')
        axes[0, 1].set_ylabel('完成率')
        axes[0, 1].set_xticks(range(len(test_names)))
        axes[0, 1].set_xticklabels(test_names, rotation=45, ha='right')
        
        # 总奖励对比
        total_rewards = [successful_results[k]['evaluation_metrics'].get('total_reward_score', 0) 
                        for k in test_names]
        
        axes[1, 0].bar(range(len(test_names)), total_rewards)
        axes[1, 0].set_title('总奖励对比')
        axes[1, 0].set_ylabel('奖励分数')
        axes[1, 0].set_xticks(range(len(test_names)))
        axes[1, 0].set_xticklabels(test_names, rotation=45, ha='right')
        
        # 目标满足率对比
        satisfied_rates = [successful_results[k]['evaluation_metrics'].get('satisfied_targets_rate', 0) 
                          for k in test_names]
        
        axes[1, 1].bar(range(len(test_names)), satisfied_rates)
        axes[1, 1].set_title('目标满足率对比')
        axes[1, 1].set_ylabel('满足率')
        axes[1, 1].set_xticks(range(len(test_names)))
        axes[1, 1].set_xticklabels(test_names, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/curves/network_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"对比图表已保存至: {self.output_dir}/curves/network_comparison.png")
    
    def save_test_data(self):
        """保存测试数据"""
        # 保存结果
        with open(f"{self.output_dir}/reports/test_results.pkl", 'wb') as f:
            pickle.dump(self.results, f)
        
        # 保存训练历史
        with open(f"{self.output_dir}/reports/training_histories.pkl", 'wb') as f:
            pickle.dump(self.training_histories, f)
        
        # 保存JSON格式的结果
        json_results = {}
        for k, v in self.results.items():
            if 'error' not in v:
                json_results[k] = {
                    'network_type': v['network_type'],
                    'scenario_name': v['scenario_name'],
                    'training_time': v['training_time'],
                    'evaluation_metrics': v['evaluation_metrics']
                }
        
        with open(f"{self.output_dir}/reports/test_results.json", 'w', encoding='utf-8') as f:
            json.dump(json_results, f, ensure_ascii=False, indent=2)
        
        print(f"测试数据已保存至: {self.output_dir}/reports/")

def main():
    """主函数"""
    tester = NetworkConvergenceTester()
    tester.run_all_tests()
    tester.save_test_data()

if __name__ == "__main__":
    main() 