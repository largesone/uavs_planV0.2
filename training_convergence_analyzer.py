# -*- coding: utf-8 -*-
# 文件名: training_convergence_analyzer.py
# 描述: 训练收敛分析器 - 自动分析训练历史，判断早熟、过拟合、欠拟合等问题

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class TrainingConvergenceAnalyzer:
    """训练收敛分析器"""
    
    def __init__(self, model_dir: str):
        """
        初始化分析器
        
        Args:
            model_dir: 模型目录路径，包含训练历史和收敛图
        """
        self.model_dir = model_dir
        self.history = None
        self.rewards = None
        self.losses = None
        self.episodes = None
        
    def load_training_history(self) -> bool:
        """加载训练历史数据"""
        # 自动查找所有以training_history开头的pkl文件
        import glob
        history_pattern = os.path.join(self.model_dir, 'training_history*.pkl')
        possible_files = glob.glob(history_pattern)
        
        if not possible_files:
            print("未找到训练历史文件，尝试从收敛图中分析...")
            return False
        
        # 按文件名排序，优先选择最新的文件
        possible_files.sort(reverse=True)
        
        for history_path in possible_files:
            try:
                with open(history_path, 'rb') as f:
                    self.history = pickle.load(f)
                filename = os.path.basename(history_path)
                print(f"成功加载训练历史: {filename}")
                return True
            except Exception as e:
                print(f"加载 {history_path} 失败: {e}")
                continue
        
        print("所有训练历史文件加载失败")
        return False
    
    def extract_data_from_history(self):
        """从训练历史中提取数据"""
        if not self.history:
            return False
            
        self.rewards = np.array(self.history.get('episode_rewards', []))
        self.losses = np.array(self.history.get('episode_losses', []))
        self.episodes = len(self.rewards)
        
        if len(self.rewards) == 0:
            print("训练历史中没有找到奖励数据")
            return False
            
        print(f"训练轮次: {self.episodes}")
        print(f"奖励数据点: {len(self.rewards)}")
        print(f"损失数据点: {len(self.losses)}")
        return True
    
    def analyze_convergence(self) -> Dict:
        """分析训练收敛情况"""
        if self.rewards is None or len(self.rewards) < 10:
            return {"error": "数据不足，无法分析"}
        
        # 计算移动平均
        window_size = min(30, len(self.rewards) // 4)
        if window_size < 5:
            window_size = 5
            
        moving_avg = np.convolve(self.rewards, np.ones(window_size)/window_size, mode='valid')
        
        # 计算关键指标
        initial_avg = np.mean(self.rewards[:window_size])
        final_avg = np.mean(self.rewards[-window_size:])
        max_avg = np.max(moving_avg)
        min_avg = np.min(moving_avg)
        
        # 计算改进幅度
        improvement_ratio = (final_avg - initial_avg) / (abs(initial_avg) + 1e-6)
        
        # 计算稳定性指标
        final_std = np.std(self.rewards[-window_size:])
        final_cv = final_std / (abs(final_avg) + 1e-6)  # 变异系数
        
        # 计算收敛速度
        convergence_speed = self._calculate_convergence_speed(moving_avg)
        
        # 判断训练状态
        diagnosis = self._diagnose_training_state(
            float(improvement_ratio),
            float(final_cv),
            float(convergence_speed),
            float(initial_avg),
            float(final_avg),
            float(max_avg)
        )
        
        return {
            "initial_avg": initial_avg,
            "final_avg": final_avg,
            "max_avg": max_avg,
            "min_avg": min_avg,
            "improvement_ratio": improvement_ratio,
            "final_std": final_std,
            "final_cv": final_cv,
            "convergence_speed": convergence_speed,
            "diagnosis": diagnosis,
            "window_size": window_size,
            "total_episodes": len(self.rewards)
        }
    
    def _calculate_convergence_speed(self, moving_avg: np.ndarray) -> float:
        """计算收敛速度"""
        if len(moving_avg) < 10:
            return 0.0
            
        # 找到奖励开始稳定增长的起始点
        threshold = np.max(moving_avg) * 0.8
        stable_start = np.where(moving_avg >= threshold)[0]
        
        if len(stable_start) == 0:
            return 0.0
            
        convergence_episode = stable_start[0]
        return convergence_episode / len(moving_avg)  # 归一化的收敛速度
    
    def _diagnose_training_state(self, improvement_ratio: float, final_cv: float, 
                                convergence_speed: float, initial_avg: float, 
                                final_avg: float, max_avg: float) -> Dict:
        """诊断训练状态"""
        diagnosis = {
            "state": "unknown",
            "issues": [],
            "suggestions": []
        }
        
        # 判断是否早熟
        if improvement_ratio < 0.1 and final_avg < max_avg * 0.8:
            diagnosis["state"] = "early_stopping"
            diagnosis["issues"].append("早熟收敛")
            diagnosis["suggestions"].extend([
                "增加训练轮次",
                "提高探索率（epsilon）",
                "调整奖励函数",
                "增加网络容量"
            ])
        
        # 判断是否欠拟合
        elif improvement_ratio < 0.2 and final_cv > 0.3:
            diagnosis["state"] = "underfitting"
            diagnosis["issues"].append("欠拟合")
            diagnosis["suggestions"].extend([
                "增加训练轮次",
                "降低学习率",
                "增加网络层数或神经元",
                "调整奖励函数",
                "增加探索率"
            ])
        
        # 判断是否过拟合（如果有验证集数据）
        elif final_avg < max_avg * 0.7 and final_cv < 0.1:
            diagnosis["state"] = "overfitting"
            diagnosis["issues"].append("可能过拟合")
            diagnosis["suggestions"].extend([
                "增加正则化",
                "减少网络容量",
                "使用早停策略",
                "增加训练数据"
            ])
        
        # 判断是否收敛良好
        elif improvement_ratio > 0.3 and final_cv < 0.2:
            diagnosis["state"] = "good_convergence"
            diagnosis["issues"].append("收敛良好")
            diagnosis["suggestions"].extend([
                "可以适当微调参数",
                "考虑增加训练轮次以获得更好结果",
                "尝试不同的网络架构"
            ])
        
        # 判断是否不稳定
        elif final_cv > 0.4:
            diagnosis["state"] = "unstable"
            diagnosis["issues"].append("训练不稳定")
            diagnosis["suggestions"].extend([
                "降低学习率",
                "增加批量大小",
                "调整奖励函数",
                "使用梯度裁剪"
            ])
        
        # 默认情况
        else:
            diagnosis["state"] = "moderate"
            diagnosis["issues"].append("训练效果一般")
            diagnosis["suggestions"].extend([
                "增加训练轮次",
                "调整学习率",
                "优化网络架构",
                "改进奖励函数"
            ])
        
        return diagnosis
    
    def plot_convergence_analysis(self, analysis_result: Dict):
        """绘制收敛分析图表"""
        if self.rewards is None:
            print("没有奖励数据，无法绘制图表")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('训练收敛分析报告', fontsize=16, fontweight='bold')
        
        # 1. 奖励曲线
        ax1 = axes[0, 0]
        episodes = range(1, len(self.rewards) + 1)
        ax1.plot(episodes, self.rewards, 'b-', alpha=0.6, label='原始奖励')
        
        # 添加移动平均线
        window_size = analysis_result.get('window_size', 30)
        if len(self.rewards) >= window_size:
            moving_avg = np.convolve(self.rewards, np.ones(window_size)/window_size, mode='valid')
            moving_episodes = range(window_size, len(self.rewards) + 1)
            ax1.plot(moving_episodes, moving_avg, 'r-', linewidth=2, label=f'{window_size}轮移动平均')
        
        ax1.set_xlabel('训练轮次')
        ax1.set_ylabel('奖励值')
        ax1.set_title('训练奖励收敛曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 损失曲线（如果有）
        ax2 = axes[0, 1]
        if self.losses is not None and len(self.losses) > 0:
            ax2.plot(episodes, self.losses, 'g-', alpha=0.6, label='训练损失')
            ax2.set_xlabel('训练轮次')
            ax2.set_ylabel('损失值')
            ax2.set_title('训练损失曲线')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, '无损失数据', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('训练损失曲线')
        
        # 3. 奖励分布
        ax3 = axes[1, 0]
        ax3.hist(self.rewards, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(analysis_result['initial_avg'], color='red', linestyle='--', 
                    label=f'初期平均: {analysis_result["initial_avg"]:.2f}')
        ax3.axvline(analysis_result['final_avg'], color='green', linestyle='--', 
                    label=f'末期平均: {analysis_result["final_avg"]:.2f}')
        ax3.set_xlabel('奖励值')
        ax3.set_ylabel('频次')
        ax3.set_title('奖励分布')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 诊断结果
        ax4 = axes[1, 1]
        diagnosis = analysis_result['diagnosis']
        ax4.axis('off')
        
        # 创建诊断文本
        diagnosis_text = f"""
诊断结果: {diagnosis['state']}

关键指标:
• 初期平均奖励: {analysis_result['initial_avg']:.2f}
• 末期平均奖励: {analysis_result['final_avg']:.2f}
• 最大平均奖励: {analysis_result['max_avg']:.2f}
• 改进幅度: {analysis_result['improvement_ratio']*100:.1f}%
• 末期变异系数: {analysis_result['final_cv']:.3f}
• 收敛速度: {analysis_result['convergence_speed']:.3f}

发现的问题:
"""
        for issue in diagnosis['issues']:
            diagnosis_text += f"• {issue}\n"
        
        diagnosis_text += "\n建议措施:\n"
        for suggestion in diagnosis['suggestions']:
            diagnosis_text += f"• {suggestion}\n"
        
        ax4.text(0.05, 0.95, diagnosis_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        
        # 保存图表
        output_path = os.path.join(self.model_dir, 'convergence_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"收敛分析图表已保存至: {output_path}")
        
        # plt.show()  # 注释掉弹窗显示
    
    def generate_report(self, analysis_result: Dict) -> str:
        """生成分析报告"""
        diagnosis = analysis_result['diagnosis']
        
        report = f"""
=== 训练收敛分析报告 ===

基本信息:
• 训练轮次: {analysis_result['total_episodes']}
• 数据窗口大小: {analysis_result['window_size']}

关键指标:
• 初期平均奖励: {analysis_result['initial_avg']:.2f}
• 末期平均奖励: {analysis_result['final_avg']:.2f}
• 最大平均奖励: {analysis_result['max_avg']:.2f}
• 改进幅度: {analysis_result['improvement_ratio']*100:.1f}%
• 末期标准差: {analysis_result['final_std']:.2f}
• 变异系数: {analysis_result['final_cv']:.3f}
• 收敛速度: {analysis_result['convergence_speed']:.3f}

诊断结果: {diagnosis['state']}

发现的问题:
"""
        for issue in diagnosis['issues']:
            report += f"• {issue}\n"
        
        report += "\n建议措施:\n"
        for suggestion in diagnosis['suggestions']:
            report += f"• {suggestion}\n"
        
        return report
    
    def run_analysis(self) -> Dict:
        """运行完整的收敛分析"""
        print("开始训练收敛分析...")
        
        # 加载训练历史
        if not self.load_training_history():
            return {"error": "无法加载训练历史数据"}
        
        # 提取数据
        if not self.extract_data_from_history():
            return {"error": "无法提取训练数据"}
        
        # 分析收敛情况
        analysis_result = self.analyze_convergence()
        
        # 生成报告
        report = self.generate_report(analysis_result)
        print(report)
        
        # 绘制分析图表
        self.plot_convergence_analysis(analysis_result)
        
        return analysis_result

def main():
    """主函数"""
    # 示例用法
    model_dir = r"output\models\预置场景（有障碍）\lr0.001_g0.98_eps0.05-0.999_upd10_bs128_phi6_phrrt0_steps1000"
    if not os.path.exists(model_dir):
        print(f"模型目录不存在: {model_dir}")
        print("请修改model_dir为实际的模型目录路径")
        return
    
    # 创建分析器并运行分析
    analyzer = TrainingConvergenceAnalyzer(model_dir)
    result = analyzer.run_analysis()
    
    if "error" in result:
        print(f"分析失败: {result['error']}")
    else:
        print("分析完成！")

if __name__ == "__main__":
    main() 