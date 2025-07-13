# -*- coding: utf-8 -*-
# 文件名: data_generator.py
# 描述: (已更新) 用于生成多无人机协同任务规划所需的测试场景数据。
#      新增功能: 支持按需生成无人机数量递增的“扫描测试”系列场景。

import numpy as np
import os
import pickle
import random
from tqdm import tqdm

# =============================================================================
# section 0: 场景生成核心配置
# -----------------------------------------------------------------------------
# 在此集中配置所有场景生成参数，便于统一修改和管理。
# =============================================================================
SCENARIO_GENERATION_CONFIG = {
    # --- 新增的扫描测试模式 ---
    'uav_sweep_5_targets': {
        'num_targets': 5,            # 固定目标数量
        'uav_range': range(4, 26),   # 无人机数量范围: 4, 5, ..., 25
        'obstacles': lambda: random.randint(3, 8) # 障碍物数量范围
    },
    'uav_sweep_10_targets': {
        'num_targets': 10,           # 固定目标数量
        'uav_range': range(6, 26),   # 无人机数量范围: 6, 7, ..., 25
        'obstacles': lambda: random.randint(5, 15) # 障碍物数量范围
    },

    # --- [新增] 总需求 > 总供给的资源紧缺场景 ---
    'resource_starvation': {
        'uavs': 8,                      # 固定无人机数量
        'targets': 12,                  # 固定目标数量 (多于无人机)
        'obstacles': lambda: random.randint(5, 10) # 障碍物数量范围
    },

    # --- [已恢复注释] 原有的标准模式 ---
    # 'specified' 模式: 生成一个固定的、用于基准测试的场景。
    'specified': {},

    # 'collaborative' 模式: 强制协同，目标需求 > 单个无人机能力。
    'collaborative': { 
        'uavs': 4, 
        'targets': 3, 
        'obstacles': 3 
    },
    
    # 'mixed' 模式: 混合场景，目标需求随机，可能需要协同，也可能单机可完成。
    'mixed': { 
        'uavs': 4, 
        'targets': 3, 
        'obstacles': 4 
    },
    
    # 'complex' 模式: 大规模复杂场景，实体数量多，完全随机。
    'complex': { 
        'uavs': lambda: random.randint(10, 50), 
        'targets': lambda: random.randint(10, 50), 
        'obstacles': lambda: random.randint(5, 20) 
    }
}

# =============================================================================
# section 1: 核心数据结构定义
# =============================================================================
class UAV:
    """(数据结构) 存储无人机的属性"""
    def __init__(self, id, position, heading, resources, max_distance, velocity_range, economic_speed):
        self.id = id; self.position = np.array(position); self.heading = heading
        self.resources = np.array(resources); self.initial_resources = np.array(resources)
        self.max_distance, self.velocity_range, self.economic_speed = max_distance, velocity_range, economic_speed

class Target:
    """(数据结构) 存储目标的属性"""
    def __init__(self, id, position, resources, value):
        self.id = id; self.position = np.array(position); self.resources = np.array(resources); self.value = value

class Obstacle:
    """(数据结构) 障碍物基类"""
    def __init__(self, tolerance):
        self.tolerance = tolerance

class CircularObstacle(Obstacle):
    """(数据结构) 存储圆形障碍物的属性"""
    def __init__(self, center, radius, tolerance):
        super().__init__(tolerance)
        self.center, self.radius = np.array(center), radius

class PolygonalObstacle(Obstacle):
    """(数据结构) 存储多边形障碍物的属性"""
    def __init__(self, vertices, tolerance):
        super().__init__(tolerance)
        self.vertices = np.array(vertices)


# =============================================================================
# section 2: 场景生成器实现
# =============================================================================
class ScenarioGenerator:
    """负责根据配置生成并保存不同类型的测试场景。"""
    def __init__(self, num_scenarios_per_mode, map_size, resource_types=2, tolerance=50.0):
        self.num_scenarios = num_scenarios_per_mode
        self.map_width, self.map_height = map_size
        self.resource_types = resource_types
        self.tolerance = tolerance
        self.configs = SCENARIO_GENERATION_CONFIG

    def _generate_specified_scenario(self):
        """生成一个固定的、用于基准测试的场景。"""
        print("生成固定的基准测试场景...")
        # 无人机总资源供给为: [310, 290]
        base_uavs = [UAV(id=1, position=np.array([0, 0]), heading=np.pi / 6, resources=np.array([80, 50]), max_distance=6000, velocity_range=(50, 150), economic_speed=100), UAV(id=2, position=np.array([5000, 0]), heading=np.pi, resources=np.array([60, 90]), max_distance=6000, velocity_range=(60, 180), economic_speed=120), UAV(id=3, position=np.array([0, 4000]), heading=-np.pi / 2, resources=np.array([100, 70]), max_distance=6000, velocity_range=(70, 200), economic_speed=150), UAV(id=4, position=np.array([5000, 4000]), heading=-np.pi / 2, resources=np.array([70, 80]), max_distance=6000, velocity_range=(50, 160), economic_speed=110)]
        
        # [修改] 调整目标资源需求，确保总需求 < 总供给
        # 原总需求为 [350, 350]，现调整为 [300, 280]
        base_targets = [
            Target(id=1, position=np.array([2000, 2000]), resources=np.array([120, 100]), value=100), 
            Target(id=2, position=np.array([3000, 2500]), resources=np.array([100, 100]), value=90), 
            Target(id=3, position=np.array([2500, 1000]), resources=np.array([80, 80]), value=80)
        ]
        
        obstacles_present = [CircularObstacle(center=(1200, 1200), radius=400, tolerance=self.tolerance), PolygonalObstacle(vertices=[(3500, 3500), (4500, 4000), (4500, 3000)], tolerance=self.tolerance), PolygonalObstacle(vertices=[(3800, 500), (4800, 800), (4500, 1500), (3500, 1200)], tolerance=self.tolerance)]
        return {'uavs': base_uavs, 'targets': base_targets, 'obstacles': obstacles_present, 'scenario_name': 'fixed_benchmark_scenario'}

    
    def _generate_uavs(self, num_uavs):
        """生成一组具有随机属性的无人机。"""
        uavs = []
        for i in range(num_uavs):
            uavs.append(UAV(id=i + 1, position=np.array([random.uniform(0, self.map_width), random.uniform(0, self.map_height)]), heading=random.uniform(0, 2 * np.pi), resources=np.array([random.randint(60, 120) for _ in range(self.resource_types)]), max_distance=self.map_width * 2, velocity_range=(50, 150 + i * 5), economic_speed=100 + i * 5))
        return uavs

    def _generate_obstacles(self, num_obstacles, targets):
        """生成一组不与目标点重叠的障碍物。"""
        obstacles = []
        for _ in range(num_obstacles):
            is_safe = False
            while not is_safe:
                if random.random() > 0.4:
                    center = (random.uniform(0, self.map_width), random.uniform(0, self.map_height)); radius = random.uniform(self.map_width * 0.05, self.map_width * 0.1)
                    if not targets or all(np.linalg.norm(np.array(center) - t.position) > radius + 250 for t in targets):
                        obstacles.append(CircularObstacle(center=center, radius=radius, tolerance=self.tolerance)); is_safe = True
                else:
                    center = (random.uniform(0, self.map_width), random.uniform(0, self.map_height)); num_verts = random.randint(3, 7); radius = random.uniform(self.map_width * 0.06, self.map_width * 0.12)
                    angles = np.sort(np.random.rand(num_verts) * 2 * np.pi); vertices = [(center[0] + radius * np.cos(a) * random.uniform(0.7, 1.3), center[1] + radius * np.sin(a) * random.uniform(0.7, 1.3)) for a in angles]
                    if not targets or all(np.linalg.norm(np.array(center) - t.position) > radius * 1.5 for t in targets):
                        obstacles.append(PolygonalObstacle(vertices=vertices, tolerance=self.tolerance)); is_safe = True
        return obstacles

    def _generate_targets(self, num_targets, uavs, mode):
        """
        (已更新和修订) 根据不同模式生成一组目标。
        新增逻辑: 为 'resource_starvation' 模式跳过需求缩减步骤。
        """
        if not uavs: return []
        
        # --- 1. 正常生成原始需求 ---
        raw_targets = []
        max_uav_resources = np.max([u.resources for u in uavs], axis=0)
        for i in range(num_targets):
            position = np.array([random.uniform(self.map_width * 0.1, self.map_width * 0.9), random.uniform(self.map_height * 0.1, self.map_height * 0.9)])
            if mode == 'collaborative':
                demand = np.array([random.uniform(1.1, 1.5) * max_res for max_res in max_uav_resources])
            elif mode == 'mixed':
                if random.random() > 0.5:
                    demand = np.array([random.uniform(1.1, 1.5) * max_res for max_res in max_uav_resources])
                else:
                    demand = np.array([random.uniform(0.5, 0.9) * max_res for max_res in max_uav_resources])
            else: # 'complex', 'sweep', 或 'resource_starvation' 模式下的目标生成
                # 为 resource_starvation 模式也使用此随机逻辑，其需求大概率会超过供给
                demand = np.array([random.uniform(0.3, 1.8) * max_res for max_res in max_uav_resources])
            
            raw_targets.append(Target(id=i + 1, position=position, resources=np.maximum(1, demand.astype(int)), value=random.randint(80, 150)))

        # --- 2. [修订] 检查并有条件地调整总需求 ---
        # 对于除 'resource_starvation' 外的所有模式，保持原有逻辑，确保总供给 >= 总需求
        if mode != 'resource_starvation':
            total_uav_supply = np.sum([u.resources for u in uavs], axis=0)
            total_target_demand = np.sum([t.resources for t in raw_targets], axis=0)
            
            # 检查是否存在任何一种资源的需求超过供给
            if np.any(total_target_demand > total_uav_supply):
                # 计算每种资源超出供给的比例
                ratios = total_target_demand / total_uav_supply
                # 找到最大的超出比例，作为全局缩放的基准
                max_ratio = np.max(ratios)
                
                # 缩放所有目标的需求，并乘以一个安全系数（如0.95），确保调整后总需求小于总供给
                scaling_factor = 0.95 / max_ratio
                for target in raw_targets:
                    adjusted_resources = (target.resources * scaling_factor).astype(int)
                    # 确保资源至少为1
                    target.resources = np.maximum(adjusted_resources, 1)

        # 如果模式是 'resource_starvation'，则直接返回原始生成的、需求可能超标的目标
        return raw_targets

    def generate_and_save(self, base_dir='scenarios'):
        """
        (已更新) 主函数，循环所有配置模式，包括新增的扫描模式，生成并保存所有场景。
        """
        print("开始生成测试场景数据...")
        for mode, cfg in self.configs.items():
            save_dir = os.path.join(base_dir, mode); os.makedirs(save_dir, exist_ok=True)
            
            # --- 新增逻辑: 处理扫描模式 ---
            if 'sweep' in mode:
                print(f"--- 正在生成 '{mode}' 类型的扫描系列场景 ---")
                num_targets = cfg['num_targets']
                for num_uavs in tqdm(cfg['uav_range'], desc=f"生成 {mode} 场景"):
                    num_obstacles = cfg['obstacles']() if callable(cfg['obstacles']) else cfg['obstacles']
                    uavs = self._generate_uavs(num_uavs)
                    # 扫描场景中的目标资源需求，我们默认按 'complex' 模式的随机方式生成
                    targets = self._generate_targets(num_targets, uavs, mode='complex')
                    obstacles = self._generate_obstacles(num_obstacles, targets)
                    
                    scenario_name = f"uavs_{num_uavs:02d}_targets_{num_targets:02d}"
                    scenario_data = {'uavs': uavs, 'targets': targets, 'obstacles': obstacles, 'scenario_name': scenario_name}
                    
                    filepath = os.path.join(save_dir, f'{scenario_name}.pkl')
                    with open(filepath, 'wb') as f: pickle.dump(scenario_data, f)
                continue # 处理完扫描模式后，进入下一个循环

            # --- 原有逻辑: 处理标准模式 ---
            if mode == 'specified':
                scenario_data = self._generate_specified_scenario()
                filepath = os.path.join(save_dir, 'fixed_benchmark_scenario.pkl')
                with open(filepath, 'wb') as f: pickle.dump(scenario_data, f)
                print(f"--- 已生成 'specified' 模式下的固定场景 ---")
                continue

            print(f"--- 正在生成 '{mode}' 类型场景 ({self.num_scenarios}个) ---")
            for i in tqdm(range(self.num_scenarios), desc=f"生成 {mode} 场景"):
                num_uavs = cfg['uavs']() if callable(cfg['uavs']) else cfg['uavs']; num_targets = cfg['targets']() if callable(cfg['targets']) else cfg['targets']; num_obstacles = cfg['obstacles']() if callable(cfg['obstacles']) else cfg['obstacles']
                uavs = self._generate_uavs(num_uavs); targets = self._generate_targets(num_targets, uavs, mode=mode); obstacles = self._generate_obstacles(num_obstacles, targets)
                scenario_data = {'uavs': uavs, 'targets': targets, 'obstacles': obstacles, 'scenario_name': f'{mode}_scenario_{i+1}'}
                filepath = os.path.join(save_dir, f'scenario_{i+1}.pkl')
                with open(filepath, 'wb') as f: pickle.dump(scenario_data, f)
        print(f"\n数据生成完成！所有场景已保存至 '{base_dir}' 文件夹。")

# =============================================================================
# section 3: 脚本执行入口
# =============================================================================
if __name__ == '__main__':
    NUM_SCENARIOS_PER_RANDOM_MODE = 10 # 为节省时间，随机模式各生成10个
    MAP_SIZE = (8000, 6000)
    
    generator = ScenarioGenerator(
        num_scenarios_per_mode=NUM_SCENARIOS_PER_RANDOM_MODE,
        map_size=MAP_SIZE
    )
    generator.generate_and_save()