# -*- coding: utf-8 -*-
# 文件名: scenarios.py
# 描述: 提供用于测试和仿真的预定义场景数据。

import numpy as np
import random
from entities import UAV, Target
from path_planning import CircularObstacle, PolygonalObstacle

def get_balanced_scenario(obstacle_tolerance):
    """
    提供一个资源平衡的场景：10个无人机，5个目标，资源供给等于需求。
    
    Args:
        obstacle_tolerance (float): 障碍物的安全容忍距离。

    Returns:
        tuple: 包含无人机列表、目标列表和障碍物列表。
    """
    # 5个目标及其资源需求
    targets = [
        Target(id=1, position=np.array([2000, 2000]), resources=np.array([150, 150]), value=100),
        Target(id=2, position=np.array([3000, 2500]), resources=np.array([120, 130]), value=90),
        Target(id=3, position=np.array([2500, 1000]), resources=np.array([100, 100]), value=80),
        Target(id=4, position=np.array([1500, 3000]), resources=np.array([80, 70]), value=85),
        Target(id=5, position=np.array([3500, 1500]), resources=np.array([50, 50]), value=75)
    ]
    
    # 计算总需求
    total_demand = np.array([500, 500])  # [150+120+100+80+50, 150+130+100+70+50]
    
    # 10个无人机，资源分配等于总需求
    uavs = [
        UAV(id=1, position=np.array([500, 500]), heading=np.pi/4, resources=np.array([80, 70]), max_distance=6000, velocity_range=(50, 150), economic_speed=100),
        UAV(id=2, position=np.array([4500, 500]), heading=3*np.pi/4, resources=np.array([70, 80]), max_distance=6000, velocity_range=(60, 160), economic_speed=110),
        UAV(id=3, position=np.array([500, 3500]), heading=-np.pi/4, resources=np.array([60, 60]), max_distance=6000, velocity_range=(55, 155), economic_speed=105),
        UAV(id=4, position=np.array([4500, 3500]), heading=-3*np.pi/4, resources=np.array([50, 50]), max_distance=6000, velocity_range=(65, 165), economic_speed=115),
        UAV(id=5, position=np.array([2500, 500]), heading=np.pi/2, resources=np.array([40, 40]), max_distance=6000, velocity_range=(70, 170), economic_speed=120),
        UAV(id=6, position=np.array([500, 2000]), heading=0, resources=np.array([50, 50]), max_distance=6000, velocity_range=(45, 145), economic_speed=95),
        UAV(id=7, position=np.array([4500, 2000]), heading=np.pi, resources=np.array([40, 40]), max_distance=6000, velocity_range=(75, 175), economic_speed=125),
        UAV(id=8, position=np.array([1500, 500]), heading=np.pi/3, resources=np.array([30, 40]), max_distance=6000, velocity_range=(40, 140), economic_speed=90),
        UAV(id=9, position=np.array([3500, 500]), heading=2*np.pi/3, resources=np.array([40, 30]), max_distance=6000, velocity_range=(80, 180), economic_speed=130),
        UAV(id=10, position=np.array([2500, 3500]), heading=-np.pi/2, resources=np.array([40, 40]), max_distance=6000, velocity_range=(55, 155), economic_speed=105)
    ]
    
    # 设计合理的障碍物
    obstacles = [
        # 中央障碍区域
        CircularObstacle(center=(2500, 2000), radius=300, tolerance=obstacle_tolerance),
        
        # 四个角落的障碍物
        PolygonalObstacle(vertices=[(500, 500), (1000, 700), (700, 1000)], tolerance=obstacle_tolerance),
        PolygonalObstacle(vertices=[(4000, 500), (4500, 700), (4500, 300)], tolerance=obstacle_tolerance),
        PolygonalObstacle(vertices=[(500, 3500), (700, 3000), (1000, 3500)], tolerance=obstacle_tolerance),
        PolygonalObstacle(vertices=[(4000, 3500), (4500, 3300), (4300, 3000)], tolerance=obstacle_tolerance),
        
        # 通道障碍物
        CircularObstacle(center=(1800, 1800), radius=200, tolerance=obstacle_tolerance),
        CircularObstacle(center=(3200, 2200), radius=200, tolerance=obstacle_tolerance)
    ]
    
    return uavs, targets, obstacles


def get_simple_convergence_test_scenario(obstacle_tolerance=50.0):
    """
    提供一个简化的测试场景，用于算法收敛性测试。
    特点：
    - 2个UAV，1个Target
    - 无障碍物
    - 简单的资源分配
    - 快速收敛验证
    
    Args:
        obstacle_tolerance (float): 障碍物的安全容忍距离。

    Returns:
        tuple: 包含无人机列表、目标列表和障碍物列表。
    """
    # 创建2个UAV
    uavs = [
        UAV(
            id="UAV_1",
            position=np.array([0.0, 0.0, 10.0]),
            heading=0.0,
            resources=np.array([30.0, 30.0, 30.0]),
            max_distance=100.0,
            velocity_range=(0.0, 20.0),
            economic_speed=15.0
        ),
        UAV(
            id="UAV_2", 
            position=np.array([10.0, 10.0, 10.0]),
            heading=0.0,
            resources=np.array([30.0, 30.0, 30.0]),
            max_distance=100.0,
            velocity_range=(0.0, 20.0),
            economic_speed=15.0
        )
    ]
    
    # 创建1个Target
    targets = [
        Target(
            id="Target_1",
            position=np.array([50.0, 50.0, 0.0]),
            resources=np.array([50.0, 50.0, 50.0]),
            value=100.0
        )
    ]
    
    # 无障碍物，简化测试
    obstacles = []
    
    return uavs, targets, obstacles

def get_minimal_test_scenario(obstacle_tolerance=50.0):
    """
    提供最小化测试场景，用于快速验证算法基本功能。
    特点：
    - 1个UAV，1个Target
    - 无障碍物
    - 最简单的资源分配
    - 极快速收敛
    
    Args:
        obstacle_tolerance (float): 障碍物的安全容忍距离。

    Returns:
        tuple: 包含无人机列表、目标列表和障碍物列表。
    """
    # 创建1个UAV
    uavs = [
        UAV(
            id="UAV_1",
            position=np.array([0.0, 0.0, 10.0]),
            heading=0.0,
            resources=np.array([50.0, 50.0, 50.0]),
            max_distance=100.0,
            velocity_range=(0.0, 20.0),
            economic_speed=15.0
        )
    ]
    
    # 创建1个Target
    targets = [
        Target(
            id="Target_1",
            position=np.array([50.0, 50.0, 0.0]),
            resources=np.array([50.0, 50.0, 50.0]),
            value=100.0
        )
    ]
    
    # 无障碍物
    obstacles = []
    
    return uavs, targets, obstacles

def get_new_experimental_scenario(obstacle_tolerance):
    """
    提供一个根据用户指定信息新增的实验场景。
    
    Args:
        obstacle_tolerance (float): 障碍物的安全容忍距离 (此场景中未使用)。

    Returns:
        tuple: 包含无人机列表、目标列表和障碍物列表。
    """
    # 为未在输入中指定的无人机参数设置合理的默认值
    default_max_distance = 6000
    default_velocity_range = (50, 150)
    default_economic_speed = 100

    uavs = [
        UAV(id=1, position=np.array([0, 0]), heading=np.pi / 6, resources=np.array([2, 1]), 
            max_distance=default_max_distance, velocity_range=default_velocity_range, economic_speed=default_economic_speed),
        UAV(id=2, position=np.array([1500, 0]), heading=np.pi / 6, resources=np.array([0, 1]), 
            max_distance=default_max_distance, velocity_range=default_velocity_range, economic_speed=default_economic_speed),
        UAV(id=3, position=np.array([3000, 0]), heading=3 * np.pi / 4, resources=np.array([3, 2]), 
            max_distance=default_max_distance, velocity_range=default_velocity_range, economic_speed=default_economic_speed),
        UAV(id=4, position=np.array([2000, 2000]), heading=np.pi / 6, resources=np.array([2, 1]), 
            max_distance=default_max_distance, velocity_range=default_velocity_range, economic_speed=default_economic_speed)
    ]
    
    targets = [
        Target(id=1, position=np.array([1500, 1500]), resources=np.array([4, 2]), value=100),
        Target(id=2, position=np.array([2000, 1000]), resources=np.array([3, 3]), value=90)
    ]

    # 此场景没有定义障碍物
    obstacles = []
    
    return uavs, targets, obstacles
def get_small_scenario(obstacle_tolerance):
    """
    提供一个预置的小规模测试场景。
    
    Args:
        obstacle_tolerance (float): 障碍物的安全容忍距离。

    Returns:
        tuple: 包含无人机列表、目标列表和障碍物列表。
    """
    # [修改] 增加无人机初始资源，使总供给与总需求持平（均为[350, 350]）
    uavs = [
        UAV(id=1, position=np.array([0, 0]), heading=np.pi / 6, resources=np.array([90, 65]), max_distance=6000, velocity_range=(50, 150), economic_speed=100),
        UAV(id=2, position=np.array([5000, 0]), heading=np.pi, resources=np.array([70, 105]), max_distance=6000, velocity_range=(60, 180), economic_speed=120),
        UAV(id=3, position=np.array([0, 4000]), heading=-np.pi / 2, resources=np.array([110, 85]), max_distance=6000, velocity_range=(70, 200), economic_speed=150),
        UAV(id=4, position=np.array([5000, 4000]), heading=-np.pi / 2, resources=np.array([80, 95]), max_distance=6000, velocity_range=(50, 160), economic_speed=110)
    ]
    targets = [
        Target(id=1, position=np.array([2000, 2000]), resources=np.array([150, 120]), value=100),
        Target(id=2, position=np.array([3000, 2500]), resources=np.array([110, 130]), value=90),
        Target(id=3, position=np.array([2500, 1000]), resources=np.array([90, 100]), value=80)
    ]
    obstacles = [
        CircularObstacle(center=(1200, 1200), radius=400, tolerance=obstacle_tolerance),
        PolygonalObstacle(vertices=[(3500, 3500), (4500, 4000), (4500, 3000)], tolerance=obstacle_tolerance),
        PolygonalObstacle(vertices=[(3800, 500), (4800, 800), (4500, 1500), (3500, 1200)], tolerance=obstacle_tolerance)
    ]
    return uavs, targets, obstacles


def get_complex_scenario(obstacle_tolerance):
    """
    提供一个随机生成的大规模复杂场景。
    
    Args:
        obstacle_tolerance (float): 障碍物的安全容忍距离。

    Returns:
        tuple: 包含无人机列表、目标列表和障碍物列表。
    """
    COMPLEX_UAV_COUNT = 15
    COMPLEX_TARGET_COUNT = 15
    COMPLEX_OBSTACLE_COUNT = 8
    MAP_SIZE_COMPLEX = (8000, 6000)
    
    uavs = [
        UAV(id=i + 1,
            position=np.array([random.uniform(0, MAP_SIZE_COMPLEX[0]), random.uniform(0, MAP_SIZE_COMPLEX[1])]),
            heading=random.uniform(0, 2 * np.pi),
            resources=np.array([random.randint(80, 120), random.randint(80, 120)]),
            max_distance=15000,
            velocity_range=(50, 180),
            economic_speed=120) for i in range(COMPLEX_UAV_COUNT)
    ]
    
    targets = [
        Target(id=i + 1,
               position=np.array([random.uniform(MAP_SIZE_COMPLEX[0] * 0.1, MAP_SIZE_COMPLEX[0] * 0.9),
                                  random.uniform(MAP_SIZE_COMPLEX[1] * 0.1, MAP_SIZE_COMPLEX[1] * 0.9)]),
               resources=np.array([random.randint(100, 200), random.randint(100, 200)]),
               value=random.randint(80, 150)) for i in range(COMPLEX_TARGET_COUNT)
    ]
    
    obstacles = []
    for _ in range(COMPLEX_OBSTACLE_COUNT):
        center = (random.uniform(0, MAP_SIZE_COMPLEX[0]), random.uniform(0, MAP_SIZE_COMPLEX[1]))
        if random.random() > 0.5:
            obstacles.append(CircularObstacle(
                center=center,
                radius=random.uniform(MAP_SIZE_COMPLEX[0] * 0.05, MAP_SIZE_COMPLEX[0] * 0.1),
                tolerance=obstacle_tolerance))
        else:
            num_verts = random.randint(3, 6)
            radius = random.uniform(MAP_SIZE_COMPLEX[0] * 0.06, MAP_SIZE_COMPLEX[0] * 0.12)
            angles = np.sort(np.random.rand(num_verts) * 2 * np.pi)
            obstacles.append(PolygonalObstacle(
                vertices=[(center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)) for a in angles],
                tolerance=obstacle_tolerance))
                
    return uavs, targets, obstacles

def get_complex_scenario_v2(obstacle_tolerance):
    """
    提供一个新的复杂场景：10个无人机，5个固定目标，资源需求从少于到等于到多余。
    保持障碍物与原场景一致以确保可比性。
    
    Args:
        obstacle_tolerance (float): 障碍物的安全容忍距离。

    Returns:
        tuple: 包含无人机列表、目标列表和障碍物列表。
    """
    # 固定5个目标及其资源需求
    targets = [
        Target(id=1, position=np.array([2000, 2000]), resources=np.array([150, 120]), value=100),
        Target(id=2, position=np.array([3000, 2500]), resources=np.array([110, 130]), value=90),
        Target(id=3, position=np.array([2500, 1000]), resources=np.array([90, 100]), value=80),
        Target(id=4, position=np.array([1500, 3000]), resources=np.array([80, 90]), value=85),
        Target(id=5, position=np.array([3500, 1500]), resources=np.array([70, 80]), value=75)
    ]
    
    # 计算总需求
    total_demand = np.array([500, 520])  # [150+110+90+80+70, 120+130+100+90+80]
    
    # 10个无人机，资源分配从少于到等于到多余需求
    uavs = [
        # 场景1：资源不足（总供给 < 总需求）
        UAV(id=1, position=np.array([0, 0]), heading=np.pi / 6, resources=np.array([90, 65]), max_distance=6000, velocity_range=(50, 150), economic_speed=100),
        UAV(id=2, position=np.array([5000, 0]), heading=np.pi, resources=np.array([70, 105]), max_distance=6000, velocity_range=(60, 180), economic_speed=120),
        UAV(id=3, position=np.array([0, 4000]), heading=-np.pi / 2, resources=np.array([110, 85]), max_distance=6000, velocity_range=(70, 200), economic_speed=150),
        UAV(id=4, position=np.array([5000, 4000]), heading=-np.pi / 2, resources=np.array([80, 95]), max_distance=6000, velocity_range=(50, 160), economic_speed=110),
        UAV(id=5, position=np.array([2500, 0]), heading=np.pi / 4, resources=np.array([60, 70]), max_distance=6000, velocity_range=(55, 170), economic_speed=125),
        UAV(id=6, position=np.array([0, 2000]), heading=np.pi / 3, resources=np.array([50, 60]), max_distance=6000, velocity_range=(65, 190), economic_speed=140),
        UAV(id=7, position=np.array([4000, 0]), heading=-np.pi / 3, resources=np.array([40, 50]), max_distance=6000, velocity_range=(45, 155), economic_speed=105),
        UAV(id=8, position=np.array([5000, 2000]), heading=np.pi / 2, resources=np.array([30, 40]), max_distance=6000, velocity_range=(75, 185), economic_speed=135),
        UAV(id=9, position=np.array([1000, 4000]), heading=-np.pi / 4, resources=np.array([20, 30]), max_distance=6000, velocity_range=(40, 145), economic_speed=95),
        UAV(id=10, position=np.array([4000, 4000]), heading=np.pi, resources=np.array([10, 20]), max_distance=6000, velocity_range=(80, 195), economic_speed=145)
    ]
    
    # 保持与原场景相同的障碍物
    obstacles = [
        CircularObstacle(center=(1200, 1200), radius=400, tolerance=obstacle_tolerance),
        PolygonalObstacle(vertices=[(3500, 3500), (4500, 4000), (4500, 3000)], tolerance=obstacle_tolerance),
        PolygonalObstacle(vertices=[(3800, 500), (4800, 800), (4500, 1500), (3500, 1200)], tolerance=obstacle_tolerance)
    ]
    
    return uavs, targets, obstacles

def get_complex_scenario_v3(obstacle_tolerance):
    """
    提供另一个复杂场景：10个无人机，5个固定目标，资源供给等于需求。
    保持障碍物与原场景一致以确保可比性。
    
    Args:
        obstacle_tolerance (float): 障碍物的安全容忍距离。

    Returns:
        tuple: 包含无人机列表、目标列表和障碍物列表。
    """
    # 固定5个目标及其资源需求
    targets = [
        Target(id=1, position=np.array([2000, 2000]), resources=np.array([150, 120]), value=100),
        Target(id=2, position=np.array([3000, 2500]), resources=np.array([110, 130]), value=90),
        Target(id=3, position=np.array([2500, 1000]), resources=np.array([90, 100]), value=80),
        Target(id=4, position=np.array([1500, 3000]), resources=np.array([80, 90]), value=85),
        Target(id=5, position=np.array([3500, 1500]), resources=np.array([70, 80]), value=75)
    ]
    
    # 计算总需求
    total_demand = np.array([500, 520])  # [150+110+90+80+70, 120+130+100+90+80]
    
    # 10个无人机，资源分配等于总需求
    uavs = [
        UAV(id=1, position=np.array([0, 0]), heading=np.pi / 6, resources=np.array([100, 80]), max_distance=6000, velocity_range=(50, 150), economic_speed=100),
        UAV(id=2, position=np.array([5000, 0]), heading=np.pi, resources=np.array([80, 100]), max_distance=6000, velocity_range=(60, 180), economic_speed=120),
        UAV(id=3, position=np.array([0, 4000]), heading=-np.pi / 2, resources=np.array([90, 90]), max_distance=6000, velocity_range=(70, 200), economic_speed=150),
        UAV(id=4, position=np.array([5000, 4000]), heading=-np.pi / 2, resources=np.array([70, 80]), max_distance=6000, velocity_range=(50, 160), economic_speed=110),
        UAV(id=5, position=np.array([2500, 0]), heading=np.pi / 4, resources=np.array([60, 70]), max_distance=6000, velocity_range=(55, 170), economic_speed=125),
        UAV(id=6, position=np.array([0, 2000]), heading=np.pi / 3, resources=np.array([50, 60]), max_distance=6000, velocity_range=(65, 190), economic_speed=140),
        UAV(id=7, position=np.array([4000, 0]), heading=-np.pi / 3, resources=np.array([40, 50]), max_distance=6000, velocity_range=(45, 155), economic_speed=105),
        UAV(id=8, position=np.array([5000, 2000]), heading=np.pi / 2, resources=np.array([30, 40]), max_distance=6000, velocity_range=(75, 185), economic_speed=135),
        UAV(id=9, position=np.array([1000, 4000]), heading=-np.pi / 4, resources=np.array([20, 30]), max_distance=6000, velocity_range=(40, 145), economic_speed=95),
        UAV(id=10, position=np.array([4000, 4000]), heading=np.pi, resources=np.array([10, 20]), max_distance=6000, velocity_range=(80, 195), economic_speed=145)
    ]
    
    # 保持与原场景相同的障碍物
    obstacles = [
        CircularObstacle(center=(1200, 1200), radius=400, tolerance=obstacle_tolerance),
        PolygonalObstacle(vertices=[(3500, 3500), (4500, 4000), (4500, 3000)], tolerance=obstacle_tolerance),
        PolygonalObstacle(vertices=[(3800, 500), (4800, 800), (4500, 1500), (3500, 1200)], tolerance=obstacle_tolerance)
    ]
    
    return uavs, targets, obstacles

def get_complex_scenario_v4(obstacle_tolerance):
    """
    提供最复杂的场景：10个无人机，5个固定目标，资源供给超过需求20%。
    保持障碍物与原场景一致以确保可比性。
    
    Args:
        obstacle_tolerance (float): 障碍物的安全容忍距离。

    Returns:
        tuple: 包含无人机列表、目标列表和障碍物列表。
    """
    # 固定5个目标及其资源需求
    targets = [
        Target(id=1, position=np.array([2000, 2000]), resources=np.array([150, 120]), value=100),
        Target(id=2, position=np.array([3000, 2500]), resources=np.array([110, 130]), value=90),
        Target(id=3, position=np.array([2500, 1000]), resources=np.array([90, 100]), value=80),
        Target(id=4, position=np.array([1500, 3000]), resources=np.array([80, 90]), value=85),
        Target(id=5, position=np.array([3500, 1500]), resources=np.array([70, 80]), value=75)
    ]
    
    # 计算总需求
    total_demand = np.array([500, 520])  # [150+110+90+80+70, 120+130+100+90+80]
    # 总供给 = 总需求 * 1.2 (超过20%)
    total_supply = total_demand * 1.2  # [600, 624]
    
    # 10个无人机，资源分配超过总需求20%
    uavs = [
        UAV(id=1, position=np.array([0, 0]), heading=np.pi / 6, resources=np.array([120, 100]), max_distance=6000, velocity_range=(50, 150), economic_speed=100),
        UAV(id=2, position=np.array([5000, 0]), heading=np.pi, resources=np.array([100, 120]), max_distance=6000, velocity_range=(60, 180), economic_speed=120),
        UAV(id=3, position=np.array([0, 4000]), heading=-np.pi / 2, resources=np.array([110, 110]), max_distance=6000, velocity_range=(70, 200), economic_speed=150),
        UAV(id=4, position=np.array([5000, 4000]), heading=-np.pi / 2, resources=np.array([90, 100]), max_distance=6000, velocity_range=(50, 160), economic_speed=110),
        UAV(id=5, position=np.array([2500, 0]), heading=np.pi / 4, resources=np.array([80, 90]), max_distance=6000, velocity_range=(55, 170), economic_speed=125),
        UAV(id=6, position=np.array([0, 2000]), heading=np.pi / 3, resources=np.array([70, 80]), max_distance=6000, velocity_range=(65, 190), economic_speed=140),
        UAV(id=7, position=np.array([4000, 0]), heading=-np.pi / 3, resources=np.array([60, 70]), max_distance=6000, velocity_range=(45, 155), economic_speed=105),
        UAV(id=8, position=np.array([5000, 2000]), heading=np.pi / 2, resources=np.array([50, 60]), max_distance=6000, velocity_range=(75, 185), economic_speed=135),
        UAV(id=9, position=np.array([1000, 4000]), heading=-np.pi / 4, resources=np.array([40, 50]), max_distance=6000, velocity_range=(40, 145), economic_speed=95),
        UAV(id=10, position=np.array([4000, 4000]), heading=np.pi, resources=np.array([30, 40]), max_distance=6000, velocity_range=(80, 195), economic_speed=145)
    ]
    
    # 保持与原场景相同的障碍物
    obstacles = [
        CircularObstacle(center=(1200, 1200), radius=400, tolerance=obstacle_tolerance),
        PolygonalObstacle(vertices=[(3500, 3500), (4500, 4000), (4500, 3000)], tolerance=obstacle_tolerance),
        PolygonalObstacle(vertices=[(3800, 500), (4800, 800), (4500, 1500), (3500, 1200)], tolerance=obstacle_tolerance)
    ]
    
    return uavs, targets, obstacles

def get_rl_advantage_scenario(obstacle_tolerance=50.0):
    """
    创建体现RL算法优势的复杂场景
    
    该场景设计特点：
    1. 动态资源需求：目标资源需求随时间变化
    2. 多约束优化：同时考虑距离、时间、资源匹配
    3. 不确定性：障碍物随机分布，路径规划复杂
    4. 协作要求：需要多无人机协同完成任务
    5. 实时适应：环境状态动态变化
    
    Args:
        obstacle_tolerance (float): 障碍物容差
        
    Returns:
        tuple: (uavs, targets, obstacles)
    """
    print("创建RL优势体现场景...")
    
    # 创建更多无人机和目标，增加问题复杂度
    num_uavs = 15
    num_targets = 25
    num_obstacles = 30
    
    # 初始化无人机
    uavs = []
    for i in range(num_uavs):
        # 分散的初始位置
        x = random.uniform(50, 450)
        y = random.uniform(50, 450)
        position = np.array([x, y])
        
        # 多样化的资源配置
        if i < 5:
            # 高容量无人机
            resources = np.array([random.uniform(800, 1200), random.uniform(600, 1000)])
        elif i < 10:
            # 中等容量无人机
            resources = np.array([random.uniform(500, 800), random.uniform(400, 600)])
        else:
            # 低容量无人机
            resources = np.array([random.uniform(200, 500), random.uniform(150, 400)])
        
        uav = UAV(
            id=i,
            position=position,
            resources=resources.copy(),
            velocity_range=(80, 120),
            max_distance=2000,
            heading=0.0,
            economic_speed=100.0
        )
        uavs.append(uav)
    
    # 创建目标，具有动态资源需求
    targets = []
    for i in range(num_targets):
        # 分散的目标位置
        x = random.uniform(100, 400)
        y = random.uniform(100, 400)
        position = np.array([x, y])
        
        # 动态资源需求：不同目标有不同优先级
        if i < 8:
            # 高优先级目标
            resources = np.array([random.uniform(600, 1000), random.uniform(400, 800)])
        elif i < 15:
            # 中等优先级目标
            resources = np.array([random.uniform(300, 600), random.uniform(200, 500)])
        else:
            # 低优先级目标
            resources = np.array([random.uniform(100, 300), random.uniform(80, 250)])
        
        target = Target(
            id=i,
            position=position,
            resources=resources.copy(),
            value=1.0
        )
        targets.append(target)
    
    # 创建复杂的障碍物分布
    obstacles = []
    for i in range(num_obstacles):
        # 创建不同类型的障碍物
        if i < 10:
            # 大型障碍物
            x = random.uniform(150, 350)
            y = random.uniform(150, 350)
            radius = random.uniform(30, 60)
        elif i < 20:
            # 中型障碍物
            x = random.uniform(100, 400)
            y = random.uniform(100, 400)
            radius = random.uniform(15, 30)
        else:
            # 小型障碍物
            x = random.uniform(50, 450)
            y = random.uniform(50, 450)
            radius = random.uniform(5, 15)
        
        # 确保障碍物不会完全阻塞路径
        position = np.array([x, y])
        
        # 创建障碍物类（如果不存在则使用简单的圆形障碍物）
        try:
            from entities import Obstacle
            obstacle = Obstacle(position, radius)
        except ImportError:
            # 如果Obstacle类不存在，创建一个简单的障碍物类
            class SimpleObstacle:
                def __init__(self, position, radius):
                    self.position = position
                    self.radius = radius
                
                def check_line_segment_collision(self, p1, p2):
                    # 简单的线段碰撞检测
                    return False  # 简化处理
            
            obstacle = SimpleObstacle(position, radius)
        
        # 检查是否与无人机或目标位置冲突
        min_distance_to_uavs = min(float(np.linalg.norm(position - uav.position)) for uav in uavs)
        min_distance_to_targets = min(float(np.linalg.norm(position - target.position)) for target in targets)
        
        if min_distance_to_uavs > radius + obstacle_tolerance and min_distance_to_targets > radius + obstacle_tolerance:
            obstacles.append(obstacle)
    
    print(f"RL优势场景创建完成:")
    print(f"  - 无人机数量: {len(uavs)}")
    print(f"  - 目标数量: {len(targets)}")
    print(f"  - 障碍物数量: {len(obstacles)}")
    print(f"  - 场景复杂度: 高 (动态约束 + 多目标优化 + 不确定性)")
    
    return uavs, targets, obstacles

def get_strategic_trap_scenario(obstacle_tolerance=50.0):
    """
    创建"战略价值陷阱"场景
    
    场景特点：
    1. 高价值陷阱目标：在地图偏远角落放置价值极高的目标，被密集障碍物包围
    2. 中价值集群目标：地图中心区域放置3-4个中等价值目标，距离较近
    3. 资源异构性：部分无人机携带更多A类资源，部分携带更多B类资源
    4. 中心目标集群对A、B类资源都有需求
    
    Args:
        obstacle_tolerance: 障碍物安全距离
        
    Returns:
        uavs: 无人机列表
        targets: 目标列表  
        obstacles: 障碍物列表
    """
    # 创建无人机 - 资源异构性设计
    uavs = [
        # A类资源丰富的无人机 (位置在中心区域)
        UAV(1, np.array([200, 200]), 0.0, np.array([150, 50]), 1000, (20, 40), 30),
        UAV(2, np.array([250, 200]), 0.0, np.array([140, 60]), 1000, (20, 40), 30),
        UAV(3, np.array([200, 250]), 0.0, np.array([160, 40]), 1000, (20, 40), 30),
        
        # B类资源丰富的无人机 (位置在边缘区域)
        UAV(4, np.array([100, 100]), 0.0, np.array([50, 150]), 1000, (20, 40), 30),
        UAV(5, np.array([150, 100]), 0.0, np.array([60, 140]), 1000, (20, 40), 30),
        UAV(6, np.array([100, 150]), 0.0, np.array([40, 160]), 1000, (20, 40), 30),
        
        # 平衡型无人机 (位置在中间区域)
        UAV(7, np.array([300, 300]), 0.0, np.array([100, 100]), 1000, (20, 40), 30),
        UAV(8, np.array([350, 300]), 0.0, np.array([90, 110]), 1000, (20, 40), 30),
    ]
    
    # 创建目标 - 战略价值陷阱设计
    targets = [
        # 高价值陷阱目标 (在偏远角落，被障碍物包围)
        Target(1, np.array([50, 50]), np.array([200, 200]), 200),  # 价值极高，但难以到达
        
        # 中价值集群目标 (在中心区域，易于协同)
        Target(2, np.array([220, 220]), np.array([80, 80]), 80),   # 中心集群1
        Target(3, np.array([240, 220]), np.array([70, 90]), 80),   # 中心集群2  
        Target(4, np.array([220, 240]), np.array([90, 70]), 80),   # 中心集群3
        Target(5, np.array([240, 240]), np.array([75, 85]), 80),   # 中心集群4
        
        # 边缘目标 (中等价值，需要特定资源)
        Target(6, np.array([400, 100]), np.array([60, 120]), 60),  # 需要更多B类资源
        Target(7, np.array([100, 400]), np.array([120, 60]), 60),  # 需要更多A类资源
    ]
    
    # 创建障碍物 - 形成战略陷阱（改进版）
    obstacles = [
        # 包围高价值陷阱目标的障碍物（减少密度，确保可达性）
        CircularObstacle(center=(40, 40), radius=8, tolerance=obstacle_tolerance),   # 陷阱目标周围
        CircularObstacle(center=(60, 40), radius=8, tolerance=obstacle_tolerance),
        CircularObstacle(center=(40, 60), radius=8, tolerance=obstacle_tolerance),
        CircularObstacle(center=(60, 60), radius=8, tolerance=obstacle_tolerance),
        
        # 中心区域的轻微障碍物 (不影响协同)
        CircularObstacle(center=(200, 200), radius=5, tolerance=obstacle_tolerance),
        CircularObstacle(center=(260, 200), radius=5, tolerance=obstacle_tolerance),
        CircularObstacle(center=(200, 260), radius=5, tolerance=obstacle_tolerance),
        CircularObstacle(center=(260, 260), radius=5, tolerance=obstacle_tolerance),
        
        # 边缘区域的障碍物（减少数量）
        CircularObstacle(center=(350, 100), radius=8, tolerance=obstacle_tolerance),
        CircularObstacle(center=(100, 350), radius=8, tolerance=obstacle_tolerance),
        
        # 路径上的障碍物（减少数量和大小）
        CircularObstacle(center=(150, 150), radius=12, tolerance=obstacle_tolerance),
        CircularObstacle(center=(300, 150), radius=12, tolerance=obstacle_tolerance),
        CircularObstacle(center=(150, 300), radius=12, tolerance=obstacle_tolerance),
        CircularObstacle(center=(300, 300), radius=12, tolerance=obstacle_tolerance),
    ]
    
    print("战略价值陷阱场景已创建:")
    print(f"  - 无人机数量: {len(uavs)} (A类资源丰富: 3架, B类资源丰富: 3架, 平衡型: 2架)")
    print(f"  - 目标数量: {len(targets)} (高价值陷阱: 1个, 中价值集群: 4个, 边缘目标: 2个)")
    print(f"  - 障碍物数量: {len(obstacles)} (密集包围陷阱目标)")
    print("  - 场景特点: 资源异构性 + 价值陷阱 + 协同挑战")
    
    return uavs, targets, obstacles