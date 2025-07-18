# -*- coding: utf-8 -*-
# 文件名: anaylysis_uavs_scans.py
# 描述: 增强版算法对比分析系统 - 支持多算法性能对比和深度分析

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# --- 1. 环境与样式设置 ---

# 设置绘图风格，并指定支持中文的字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
    sns.set_style("whitegrid", {"font.sans-serif": ['SimHei', 'Arial']})
    print("中文字体'SimHei'设置成功。")
except Exception as e:
    print(f"中文字体设置失败，可能需要您手动安装'SimHei'字体或在代码中更换为其他字体。错误: {e}")

class AlgorithmAnalyzer:
    """增强版算法分析器"""
    
    def __init__(self):
        self.output_dir = "output/algorithm_analysis"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 定义算法分类
        self.algorithm_categories = {
            'RL': ['RL'],
            'GA': ['GA', 'GA_SC'],
            'Greedy': ['Greedy', 'Greedy_SC'],
            'All': ['RL', 'GA', 'Greedy', 'GA_SC', 'Greedy_SC']
        }
        
        # 定义性能指标
        self.performance_metrics = {
            'total_reward_score': '综合评分',
            'satisfied_targets_rate': '目标满足率',
            'resource_utilization_rate': '资源利用率',
            'load_balance_score': '负载均衡性',
            'total_time': '执行时间',
            'completion_rate': '任务完成率',
            'sync_feasibility_rate': '同步可行性'
        }

    def load_and_process_data(self) -> Optional[pd.DataFrame]:
        """加载并处理数据"""
        # 尝试加载多个可能的数据文件
        possible_files = [
            'batch_test_results_enhanced.csv',
            'batch_test_results.csv',
            'quick_param_test_results.csv',
            'param_test_results.csv'
        ]
        
        df = None
        for file in possible_files:
            try:
                df = pd.read_csv(file)
                print(f"成功加载数据文件: {file}")
                print(f"数据形状: {df.shape}")
                print(f"列名: {list(df.columns)}")
                break
            except FileNotFoundError:
                continue
        
        if df is None:
            print("错误：未找到任何数据文件。")
            print("请确保以下文件之一存在：")
            for file in possible_files:
                print(f"  - {file}")
            return None
        
        # 数据预处理
        df = self.preprocess_data(df)
        return df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据预处理"""
        # 添加时间戳列（如果没有）
        if 'timestamp' not in df.columns:
            df['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 添加场景规模列
        if 'scenario_size' not in df.columns:
            df['scenario_size'] = df['num_uavs'] + df['num_targets']
        
        # 处理缺失值
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean())
        
        # 添加算法类别
        df['algorithm_category'] = df['solver'].apply(self.categorize_algorithm)
        
        return df

    def categorize_algorithm(self, solver_name: str) -> str:
        """对算法进行分类"""
        if 'RL' in solver_name:
            return 'RL'
        elif 'GA' in solver_name:
            return 'GA'
        elif 'Greedy' in solver_name:
            return 'Greedy'
        else:
            return 'Other'

    def analyze_overall_performance(self, df: pd.DataFrame) -> Dict:
        """分析整体性能"""
        print("\n=== 整体性能分析 ===")
        
        # 基础统计
        total_tests = len(df)
        successful_tests = len(df[df['success'] == True]) if 'success' in df.columns else total_tests
        success_rate = successful_tests / total_tests * 100
        
        print(f"总测试数: {total_tests}")
        print(f"成功测试数: {successful_tests}")
        print(f"成功率: {success_rate:.2f}%")
        
        # 各算法基础统计
        if 'solver' in df.columns:
            solver_stats = df.groupby('solver').agg({
                'total_reward_score': ['count', 'mean', 'std'],
                'total_time': 'mean',
                'satisfied_targets_rate': 'mean'
            }).round(4)
            
            print("\n各算法统计:")
            for solver in solver_stats.index:
                print(f"\n{solver}:")
                print(f"  - 测试次数: {solver_stats.loc[solver, ('total_reward_score', 'count')]}")
                print(f"  - 平均评分: {solver_stats.loc[solver, ('total_reward_score', 'mean')]:.2f} ± {solver_stats.loc[solver, ('total_reward_score', 'std')]:.2f}")
                print(f"  - 平均时间: {solver_stats.loc[solver, ('total_time', 'mean')]:.2f}秒")
                print(f"  - 平均满足率: {solver_stats.loc[solver, ('satisfied_targets_rate', 'mean')]:.4f}")
        
        return {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': success_rate,
            'solver_stats': solver_stats if 'solver' in df.columns else None
        }

    def analyze_rl_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        """分析RL算法的性能表现"""
        print("\n=== RL算法性能分析 ===")
        
        # 筛选RL算法的数据
        rl_data = df[df['solver'].str.contains('RL', case=False, na=False)].copy()
        
        if rl_data.empty:
            print("未找到RL算法的数据")
            return pd.DataFrame()
        
        print(f"找到 {len(rl_data)} 条RL算法数据")
        
        # 1. 基础统计分析
        print("\n1. 基础统计分析:")
        print(f"   - 平均综合评分: {rl_data['total_reward_score'].mean():.2f}")
        print(f"   - 平均目标满足率: {rl_data['satisfied_targets_rate'].mean():.4f}")
        print(f"   - 平均资源利用率: {rl_data['resource_utilization_rate'].mean():.4f}")
        print(f"   - 平均训练时间: {rl_data['total_time'].mean():.2f}秒")
        
        # 2. 配置参数影响分析
        if 'config_name' in rl_data.columns:
            print("\n2. 配置参数影响分析:")
            config_analysis = rl_data.groupby('config_name').agg({
                'total_reward_score': ['mean', 'std'],
                'total_time': 'mean',
                'satisfied_targets_rate': 'mean'
            }).round(4)
            print(config_analysis)
        
        # 3. 场景复杂度影响
        print("\n3. 场景复杂度影响:")
        if 'scenario_size' in rl_data.columns:
            size_analysis = rl_data.groupby('scenario_size').agg({
                'total_reward_score': 'mean',
                'total_time': 'mean',
                'satisfied_targets_rate': 'mean'
            }).round(4)
            print(size_analysis)
        
        return rl_data

    def create_comprehensive_plots(self, df: pd.DataFrame):
        """创建综合分析图表"""
        if df.empty:
            print("没有数据可以绘图")
            return
        
        # 1. 算法性能对比图
        self.plot_algorithm_comparison(df)
        
        # 2. RL算法详细分析
        rl_data = self.analyze_rl_performance(df)
        if not rl_data.empty:
            self.plot_rl_analysis(rl_data)
        
        # 3. 场景复杂度分析
        self.plot_scenario_complexity_analysis(df)
        
        # 4. 参数影响分析
        self.plot_parameter_impact_analysis(df)
        
        # 5. 性能雷达图
        self.plot_performance_radar(df)

    def plot_algorithm_comparison(self, df: pd.DataFrame):
        """绘制算法性能对比图"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('算法性能综合对比分析', fontsize=16)
        
        # 筛选成功的测试
        success_df = df[df['success'] == True] if 'success' in df.columns else df
        
        if success_df.empty:
            print("没有成功的测试数据")
            return
        
        # 1. 综合评分对比
        ax1 = axes[0, 0]
        solver_scores = success_df.groupby('solver')['total_reward_score'].agg(['mean', 'std']).sort_values('mean', ascending=False)
        ax1.bar(solver_scores.index, solver_scores['mean'], yerr=solver_scores['std'], capsize=5, alpha=0.7)
        ax1.set_title('算法综合评分对比')
        ax1.set_ylabel('综合评分')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 2. 目标满足率对比
        ax2 = axes[0, 1]
        solver_satisfaction = success_df.groupby('solver')['satisfied_targets_rate'].mean().sort_values(ascending=False)
        colors = plt.cm.Set3(np.linspace(0, 1, len(solver_satisfaction)))
        ax2.bar(solver_satisfaction.index, solver_satisfaction.values, color=colors)
        ax2.set_title('目标满足率对比')
        ax2.set_ylabel('满足率')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. 资源利用率对比
        ax3 = axes[0, 2]
        solver_utilization = success_df.groupby('solver')['resource_utilization_rate'].mean().sort_values(ascending=False)
        ax3.bar(solver_utilization.index, solver_utilization.values, color=colors)
        ax3.set_title('资源利用率对比')
        ax3.set_ylabel('利用率')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. 执行时间对比
        ax4 = axes[1, 0]
        solver_time = success_df.groupby('solver')['total_time'].mean().sort_values(ascending=True)
        ax4.bar(solver_time.index, solver_time.values, color=colors)
        ax4.set_title('执行时间对比')
        ax4.set_ylabel('时间(秒)')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # 5. 负载均衡性对比
        ax5 = axes[1, 1]
        solver_balance = success_df.groupby('solver')['load_balance_score'].mean().sort_values(ascending=False)
        ax5.bar(solver_balance.index, solver_balance.values, color=colors)
        ax5.set_title('负载均衡性对比')
        ax5.set_ylabel('均衡性评分')
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True, alpha=0.3)
        
        # 6. 成功率对比
        ax6 = axes[1, 2]
        if 'success' in df.columns:
            solver_success = df.groupby('solver')['success'].mean().sort_values(ascending=False)
            ax6.bar(solver_success.index, solver_success.values, color=colors)
            ax6.set_title('算法成功率对比')
            ax6.set_ylabel('成功率')
            ax6.tick_params(axis='x', rotation=45)
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, '无成功率数据', ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('成功率对比')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'algorithm_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()

    def plot_rl_analysis(self, rl_data: pd.DataFrame):
        """绘制RL算法详细分析"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('RL算法详细分析', fontsize=16)
        
        # 1. 配置参数影响
        if 'config_name' in rl_data.columns:
            ax1 = axes[0, 0]
            config_scores = rl_data.groupby('config_name')['total_reward_score'].agg(['mean', 'std'])
            ax1.bar(config_scores.index, config_scores['mean'], yerr=config_scores['std'], capsize=5)
            ax1.set_title('配置对综合评分的影响')
            ax1.set_ylabel('综合评分')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
        
        # 2. 训练时间分析
        ax2 = axes[0, 1]
        if 'config_name' in rl_data.columns:
            config_time = rl_data.groupby('config_name')['total_time'].mean()
            ax2.bar(config_time.index, config_time.values)
            ax2.set_title('配置对执行时间的影响')
            ax2.set_ylabel('时间(秒)')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
        
        # 3. 场景规模影响
        ax3 = axes[1, 0]
        if 'scenario_size' in rl_data.columns:
            ax3.scatter(rl_data['scenario_size'], rl_data['total_reward_score'], alpha=0.6)
            ax3.set_xlabel('场景规模(UAV数+目标数)')
            ax3.set_ylabel('综合评分')
            ax3.set_title('场景规模对评分的影响')
            ax3.grid(True, alpha=0.3)
        
        # 4. 障碍物模式影响
        ax4 = axes[1, 1]
        if 'obstacle_mode' in rl_data.columns:
            obstacle_impact = rl_data.groupby('obstacle_mode')['total_reward_score'].mean()
            ax4.bar(obstacle_impact.index, obstacle_impact.values)
            ax4.set_title('障碍物模式对评分的影响')
            ax4.set_ylabel('综合评分')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'rl_detailed_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()

    def plot_scenario_complexity_analysis(self, df: pd.DataFrame):
        """绘制场景复杂度分析"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('场景复杂度分析', fontsize=16)
        
        # 1. 场景规模分布
        ax1 = axes[0, 0]
        if 'scenario_size' in df.columns:
            ax1.hist(df['scenario_size'], bins=20, alpha=0.7, edgecolor='black')
            ax1.set_xlabel('场景规模')
            ax1.set_ylabel('测试次数')
            ax1.set_title('场景规模分布')
            ax1.grid(True, alpha=0.3)
        
        # 2. 成功率与场景规模关系
        ax2 = axes[0, 1]
        if 'scenario_size' in df.columns and 'success' in df.columns:
            size_success = df.groupby('scenario_size')['success'].mean()
            ax2.plot(size_success.index, size_success.values, 'o-')
            ax2.set_xlabel('场景规模')
            ax2.set_ylabel('成功率')
            ax2.set_title('成功率与场景规模关系')
            ax2.grid(True, alpha=0.3)
        
        # 3. 各算法在不同规模场景下的表现
        ax3 = axes[1, 0]
        if 'scenario_size' in df.columns:
            for solver in df['solver'].unique():
                solver_data = df[df['solver'] == solver]
                ax3.scatter(solver_data['scenario_size'], solver_data['total_reward_score'], 
                           alpha=0.6, label=solver)
            ax3.set_xlabel('场景规模')
            ax3.set_ylabel('综合评分')
            ax3.set_title('各算法在不同规模场景下的表现')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. 执行时间与场景规模关系
        ax4 = axes[1, 1]
        if 'scenario_size' in df.columns:
            for solver in df['solver'].unique():
                solver_data = df[df['solver'] == solver]
                ax4.scatter(solver_data['scenario_size'], solver_data['total_time'], 
                           alpha=0.6, label=solver)
            ax4.set_xlabel('场景规模')
            ax4.set_ylabel('执行时间(秒)')
            ax4.set_title('执行时间与场景规模关系')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'scenario_complexity_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()

    def plot_parameter_impact_analysis(self, df: pd.DataFrame):
        """绘制参数影响分析"""
        # 筛选RL算法的数据
        rl_df = df[df['solver'].str.contains('RL', case=False, na=False)].copy()
        
        if rl_df.empty:
            print("没有RL算法的数据，跳过参数影响分析")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('参数影响分析', fontsize=16)
        
        # 1. 训练轮次影响
        if 'EPISODES' in rl_df.columns:
            ax1 = axes[0, 0]
            episodes_group = rl_df.groupby('EPISODES').mean()
            ax1.plot(episodes_group.index, episodes_group['total_reward_score'], 'o-', label='综合评分')
            ax1.set_xlabel('训练轮次')
            ax1.set_ylabel('综合评分')
            ax1.set_title('训练轮次对评分的影响')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        
        # 2. PH-RRT参数影响
        if 'USE_PHRRT_DURING_TRAINING' in rl_df.columns:
            ax2 = axes[0, 1]
            phrrt_group = rl_df.groupby('USE_PHRRT_DURING_TRAINING').mean()
            x_pos = [0, 1]
            ax2.bar(x_pos, phrrt_group['total_reward_score'], 
                    tick_label=['快速近似', '高精度PH-RRT'], color=['lightblue', 'lightcoral'])
            ax2.set_ylabel('综合评分')
            ax2.set_title('距离计算方式对评分的影响')
            ax2.grid(True, alpha=0.3)
        
        # 3. 学习率影响
        if 'LEARNING_RATE' in rl_df.columns:
            ax3 = axes[1, 0]
            lr_group = rl_df.groupby('LEARNING_RATE').mean()
            ax3.plot(lr_group.index, lr_group['total_reward_score'], 's-', color='green')
            ax3.set_xlabel('学习率')
            ax3.set_ylabel('综合评分')
            ax3.set_title('学习率对评分的影响')
            ax3.grid(True, alpha=0.3)
        
        # 4. 批次大小影响
        if 'BATCH_SIZE' in rl_df.columns:
            ax4 = axes[1, 1]
            batch_group = rl_df.groupby('BATCH_SIZE').mean()
            ax4.plot(batch_group.index, batch_group['total_reward_score'], '^-', color='purple')
            ax4.set_xlabel('批次大小')
            ax4.set_ylabel('综合评分')
            ax4.set_title('批次大小对评分的影响')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'parameter_impact_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()

    def plot_performance_radar(self, df: pd.DataFrame):
        """绘制性能雷达图"""
        # 筛选成功的测试
        success_df = df[df['success'] == True] if 'success' in df.columns else df
        
        if success_df.empty:
            print("没有成功的测试数据，跳过雷达图")
            return
        
        # 为每个算法计算平均性能指标
        solvers = success_df['solver'].unique()
        
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
        
        # 定义性能指标
        metrics = ['satisfied_targets_rate', 'resource_utilization_rate', 'load_balance_score', 'sync_feasibility_rate']
        metric_names = ['目标满足率', '资源利用率', '负载均衡性', '同步可行性']
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(solvers)))
        
        for i, solver in enumerate(solvers):
            solver_data = success_df[success_df['solver'] == solver]
            
            # 计算归一化指标
            values = []
            for metric in metrics:
                if metric in solver_data.columns:
                    # 归一化到0-1范围
                    max_val = success_df[metric].max()
                    min_val = success_df[metric].min()
                    if max_val > min_val:
                        normalized_val = (solver_data[metric].mean() - min_val) / (max_val - min_val)
                    else:
                        normalized_val = 0.5
                    values.append(normalized_val)
                else:
                    values.append(0.5)
            
            values += values[:1]  # 闭合图形
            
            ax.plot(angles, values, 'o-', linewidth=2, label=solver, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names)
        ax.set_ylim(0, 1)
        ax.set_title('算法性能雷达图', size=16, y=1.08)
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_radar.png'), dpi=300, bbox_inches='tight')
        plt.show()

    def generate_comprehensive_report(self, df: pd.DataFrame):
        """生成综合分析报告"""
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("算法对比分析综合报告")
        report_lines.append("="*80)
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 1. 数据概览
        report_lines.append("1. 数据概览")
        report_lines.append("-"*40)
        report_lines.append(f"总数据条数: {len(df)}")
        report_lines.append(f"算法种类: {df['solver'].nunique()}")
        report_lines.append(f"场景数量: {df['scenario_name'].nunique() if 'scenario_name' in df.columns else '未知'}")
        report_lines.append(f"数据时间范围: {df['timestamp'].min() if 'timestamp' in df.columns else '未知'} 到 {df['timestamp'].max() if 'timestamp' in df.columns else '未知'}")
        report_lines.append("")
        
        # 2. 算法性能排名
        report_lines.append("2. 算法性能排名")
        report_lines.append("-"*40)
        success_df = df[df['success'] == True] if 'success' in df.columns else df
        
        if not success_df.empty:
            solver_ranking = success_df.groupby('solver').agg({
                'total_reward_score': ['mean', 'std'],
                'satisfied_targets_rate': 'mean',
                'resource_utilization_rate': 'mean',
                'total_time': 'mean',
                'load_balance_score': 'mean'
            }).round(4)
            
            # 按综合评分排序
            solver_ranking = solver_ranking.sort_values(('total_reward_score', 'mean'), ascending=False)
            
            for solver in solver_ranking.index:
                report_lines.append(f"\n{solver}算法 (排名 {list(solver_ranking.index).index(solver) + 1}):")
                report_lines.append(f"  - 平均综合评分: {solver_ranking.loc[solver, ('total_reward_score', 'mean')]:.2f} ± {solver_ranking.loc[solver, ('total_reward_score', 'std')]:.2f}")
                report_lines.append(f"  - 平均目标满足率: {solver_ranking.loc[solver, ('satisfied_targets_rate', 'mean')]:.4f}")
                report_lines.append(f"  - 平均资源利用率: {solver_ranking.loc[solver, ('resource_utilization_rate', 'mean')]:.4f}")
                report_lines.append(f"  - 平均执行时间: {solver_ranking.loc[solver, ('total_time', 'mean')]:.2f}秒")
                report_lines.append(f"  - 平均负载均衡性: {solver_ranking.loc[solver, ('load_balance_score', 'mean')]:.4f}")
        
        # 3. 算法类别分析
        report_lines.append("\n3. 算法类别分析")
        report_lines.append("-"*40)
        if 'algorithm_category' in df.columns:
            category_analysis = df.groupby('algorithm_category').agg({
                'total_reward_score': 'mean',
                'total_time': 'mean',
                'satisfied_targets_rate': 'mean'
            }).round(4)
            
            for category in category_analysis.index:
                report_lines.append(f"\n{category}类算法:")
                report_lines.append(f"  - 平均综合评分: {category_analysis.loc[category, 'total_reward_score']:.2f}")
                report_lines.append(f"  - 平均执行时间: {category_analysis.loc[category, 'total_time']:.2f}秒")
                report_lines.append(f"  - 平均目标满足率: {category_analysis.loc[category, 'satisfied_targets_rate']:.4f}")
        
        # 4. 场景复杂度影响
        report_lines.append("\n4. 场景复杂度影响分析")
        report_lines.append("-"*40)
        if 'scenario_size' in df.columns:
            size_analysis = df.groupby('scenario_size').agg({
                'success': 'mean' if 'success' in df.columns else lambda x: 1,
                'total_reward_score': 'mean',
                'total_time': 'mean'
            }).round(4)
            
            for size in size_analysis.index:
                report_lines.append(f"\n场景规模 {size} (UAV+目标):")
                if 'success' in df.columns:
                    report_lines.append(f"  - 成功率: {size_analysis.loc[size, 'success']:.4f}")
                report_lines.append(f"  - 平均综合评分: {size_analysis.loc[size, 'total_reward_score']:.2f}")
                report_lines.append(f"  - 平均执行时间: {size_analysis.loc[size, 'total_time']:.2f}秒")
        
        # 5. 配置参数影响（针对RL算法）
        report_lines.append("\n5. RL算法配置参数影响")
        report_lines.append("-"*40)
        rl_df = df[df['solver'].str.contains('RL', case=False, na=False)]
        if not rl_df.empty and 'config_name' in rl_df.columns:
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
        
        # 6. 错误分析
        report_lines.append("\n6. 错误分析")
        report_lines.append("-"*40)
        if 'success' in df.columns:
            failed_tests = df[df['success'] == False]
            if not failed_tests.empty and 'error_message' in failed_tests.columns:
                error_counts = failed_tests['error_message'].value_counts()
                for error, count in error_counts.head(10).items():
                    report_lines.append(f"  - {error}: {count}次")
            else:
                report_lines.append("  无失败测试或错误信息")
        else:
            report_lines.append("  无成功率数据")
        
        # 7. 结论和建议
        report_lines.append("\n7. 结论和建议")
        report_lines.append("-"*40)
        
        if not success_df.empty:
            best_solver = solver_ranking.index[0]
            report_lines.append(f"  - 最佳综合性能算法: {best_solver}")
            report_lines.append(f"  - 平均评分: {solver_ranking.loc[best_solver, ('total_reward_score', 'mean')]:.2f}")
            
            fastest_solver = success_df.groupby('solver')['total_time'].mean().idxmin()
            report_lines.append(f"  - 最快执行算法: {fastest_solver}")
            report_lines.append(f"  - 平均时间: {success_df.groupby('solver')['total_time'].mean().min():.2f}秒")
            
            most_satisfied_solver = success_df.groupby('solver')['satisfied_targets_rate'].mean().idxmax()
            report_lines.append(f"  - 最高目标满足率算法: {most_satisfied_solver}")
            report_lines.append(f"  - 满足率: {success_df.groupby('solver')['satisfied_targets_rate'].mean().max():.4f}")
        
        # 保存报告
        report_path = os.path.join(self.output_dir, 'comprehensive_analysis_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"综合分析报告已保存到: {report_path}")

def main():
    """主函数"""
    print("开始增强版算法对比分析...")
    
    analyzer = AlgorithmAnalyzer()
    
    # 加载数据
    df = analyzer.load_and_process_data()
    if df is None:
        return
    
    # 整体性能分析
    overall_stats = analyzer.analyze_overall_performance(df)
    
    # 创建综合分析图表
    analyzer.create_comprehensive_plots(df)
    
    # 生成综合报告
    analyzer.generate_comprehensive_report(df)
    
    print(f"\n分析完成！所有结果已保存到: {analyzer.output_dir}")
    print("="*60)
    print("生成的文件包括:")
    print("  - algorithm_comparison.png: 算法性能对比图")
    print("  - rl_detailed_analysis.png: RL算法详细分析")
    print("  - scenario_complexity_analysis.png: 场景复杂度分析")
    print("  - parameter_impact_analysis.png: 参数影响分析")
    print("  - performance_radar.png: 性能雷达图")
    print("  - comprehensive_analysis_report.txt: 综合分析报告")
    print("="*60)

if __name__ == "__main__":
    main()