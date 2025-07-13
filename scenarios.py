# -*- coding: utf-8 -*-
# 文件名: scenarios.py
# 描述: 提供用于测试和仿真的预定义场景数据。

import numpy as np
import random
from entities import UAV, Target
from path_planning import CircularObstacle, PolygonalObstacle
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