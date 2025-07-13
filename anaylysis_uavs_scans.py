# 导入所需库
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 1. 环境与样式设置 ---

# 设置绘图风格，并指定支持中文的字体
# 如果您的系统中没有'SimHei'字体，请替换为其他已安装的中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
    sns.set_style("whitegrid", {"font.sans-serif": ['SimHei', 'Arial']})
    print("中文字体'SimHei'设置成功。")
except Exception as e:
    print(f"中文字体设置失败，可能需要您手动安装'SimHei'字体或在代码中更换为其他字体。错误: {e}")


# --- 2. 数据加载与处理 ---

# 从CSV文件加载数据
try:
    df = pd.read_csv('batch_test_results_with_new_scores_v2.csv')
    print("成功加载数据文件。")
except FileNotFoundError:
    print("错误：未找到'batch_test_results_with_new_scores_v2.csv'文件。")
    print("请确保数据文件与此脚本位于同一目录下。")
    exit() # 如果文件不存在，则退出脚本

# 筛选出本次分析所需的，以'uav_'开头的场景
df_uav = df[df['scenario'].str.startswith('uavs_', na=False)].copy()

# 自动识别场景中的目标数量分组（通常只有一个，但代码有扩展性）
target_groups = df_uav['num_targets'].unique()

# 为每个算法定义独特的线条样式、标记和颜色，以便在图表中清晰区分
style_map = {
    'RL':        {'line': '-', 'marker': 'o', 'color': 'red'},
    'GA':        {'line': '--', 'marker': 's', 'color': 'blue'},
    'Greedy':    {'line': ':', 'marker': '^', 'color': 'green'},
    'GA_SC':     {'line': '-.', 'marker': 'D', 'color': 'purple'},
    'Greedy_SC': {'line': (0, (3, 1, 1, 1)), 'marker': 'x', 'color': 'orange'}
}


# --- 3. 循环绘图 ---

# 遍历每个目标数量分组（例如，所有10个目标的场景）
for num_t in target_groups:
    print(f"--- 正在为 {num_t} 个目标的场景生成图表 ---")
    df_group = df_uav[df_uav['num_targets'] == num_t].copy()

    # 按求解器(solver)和无人机数量(num_uavs)对数据进行分组
    # 然后计算每个组内各项评估指标的平均值
    agg_data = df_group.groupby(['solver', 'num_uavs']).agg(
        avg_score=('new_evaluation_score_v2', 'mean'),
        avg_time=('total_time', 'mean'),
        avg_completion=('completion_rate', 'mean')
    ).reset_index()

    # 创建一个包含3个子图的图窗
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    fig.suptitle(f'固定目标数量 ({num_t}个) 时，算法性能随无人机数量变化对比', fontsize=18, y=0.95)

    # --- 绘制子图1：解质量 (综合得分) ---
    ax1 = axes[0]
    for solver_name, style in style_map.items():
        solver_data = agg_data[agg_data['solver'] == solver_name]
        if not solver_data.empty:
            ax1.plot(solver_data['num_uavs'], solver_data['avg_score'],
                     linestyle=style['line'], marker=style['marker'],
                     label=solver_name, color=style['color'], lw=2) # lw=2 加粗线条
    ax1.set_ylabel('解质量 (综合得分)', fontsize=12)
    ax1.set_title('解质量对比分析', fontsize=14)
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--')

    # --- 绘制子图2：求解效率 (总耗时) ---
    ax2 = axes[1]
    for solver_name, style in style_map.items():
        solver_data = agg_data[agg_data['solver'] == solver_name]
        if not solver_data.empty:
            ax2.plot(solver_data['num_uavs'], solver_data['avg_time'],
                     linestyle=style['line'], marker=style['marker'],
                     label=solver_name, color=style['color'], lw=2)
    ax2.set_ylabel('求解效率 (总耗时 - 秒)', fontsize=12)
    ax2.set_title('求解效率对比分析', fontsize=14)
    ax2.set_yscale('log')  # 时间跨度大，使用对数坐标轴能更好地展示数量级差异
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--')

    # --- 绘制子图3：任务完成率 ---
    ax3 = axes[2]
    for solver_name, style in style_map.items():
        solver_data = agg_data[agg_data['solver'] == solver_name]
        if not solver_data.empty:
            ax3.plot(solver_data['num_uavs'], solver_data['avg_completion'],
                     linestyle=style['line'], marker=style['marker'],
                     label=solver_name, color=style['color'], lw=2)
    ax3.set_xlabel('无人机数量', fontsize=12)
    ax3.set_ylabel('任务完成率', fontsize=12)
    ax3.set_title('任务完成率对比分析', fontsize=14)
    ax3.legend()
    ax3.grid(True, which='both', linestyle='--')
    ax3.set_ylim(0, 1.1)  # 设置Y轴范围为0%到110%，留出顶部空间

    # 调整布局并显示图表
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # 调整子图布局以适应主标题
    
    # 可以选择保存图表到文件
    plt.savefig(f"analysis_chart_targets_{num_t}.png", dpi=300)
    
    plt.show()

print("\n代码执行完毕。")