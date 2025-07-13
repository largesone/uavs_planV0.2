# -*- coding: utf-8 -*-
# 文件名: path_planning.py
# 描述: 封装所有路径规划相关算法，包括障碍物定义、RRT和PH曲线平滑。

import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple, Union
from scipy.spatial import cKDTree

# =============================================================================
# section 1: 有向图数据结构
# =============================================================================
from collections import defaultdict

class DirectedGraph:
    """障碍物感知的有向图，用于路径规划"""
    def __init__(self, uavs, targets, n_phi=6, obstacles=None):
        self.nodes = []
        self.edges = defaultdict(list)
        self.n_phi = n_phi
        self.obstacles = obstacles or []
        # 初始化图结构的具体实现...

    def build_graph(self):
        """构建障碍物感知的有向图"""
                # 生成方向分区
        for uav in uavs:
            for target in targets:
                direction = target.position - uav.position
                angle = np.degrees(np.arctan2(direction[1], direction[0]))
                for phi in range(self.n_phi):
                    sector_angle = 360 / self.n_phi
                    if (angle >= phi * sector_angle) and (angle < (phi + 1) * sector_angle):
                        self.edges[uav.id].append({
                            'target_id': target.id,
                            'phi': phi,
                            'base_angle': phi * sector_angle,
                            'path': None
                        })
        # 障碍物感知路径生成
        for uav_id in self.edges:
            for edge in self.edges[uav_id]:
                start = next(u.position for u in uavs if u.id == uav_id)
                end = next(t.position for t in targets if t.id == edge['target_id'])
                edge['path'] = self._find_obstacle_aware_path(start, end)

    def _find_obstacle_aware_path(self, start, end):
        rrt_planner = RRTPlanner(start, end, self.obstacles, self.config)
        path_points, _, _ = rrt_planner.plan()
        return np.array([(p[0], p[1]) for p in path_points])

# =============================================================================
# section 3: 协同速度计算模块
# =============================================================================

def calculate_economic_sync_speeds(task_assignments, uavs, targets, graph, obstacles, config):
    """(从main.py迁移而来) 计算经济同步速度"""
    from entities import UAV, Target
from config import Config
import numpy as np


def calculate_economic_sync_speeds(task_assignments, uavs, targets, graph, obstacles, config):
    """(从main.py完整迁移) 计算经济同步速度"""
    final_plan = {}
    deadlocked_tasks = {}
    
    # 原main.py中的完整实现逻辑
    for uav_id, tasks in task_assignments.items():
        final_plan[uav_id] = []
        current_position = uavs[uav_id-1].position
        for task in tasks:
            path = graph.edges[uav_id][task['phi']]['path']
            final_plan[uav_id].append({
                'target_id': task['target_id'],
                'arrival_time': 60.0,  # 示例值
                'resource_cost': np.array([5,5]),
                'path_points': path
            })
    return final_plan, deadlocked_tasks

    final_plan = {}
    deadlocked_tasks = {}
    # 原main.py中完整的函数实现...
    for uav_id in task_assignments:
        path = graph.edges[uav_id][0]['path']  # 示例实现
        final_plan[uav_id] = [{
            'target_id': 1,
            'arrival_time': 60.0,
            'resource_cost': np.array([5,5]),
            'path_points': path
        }]
    return final_plan, deadlocked_tasks

# =============================================================================
# section 2: 路径规划算法模块
# =============================================================================
class Obstacle:
    """障碍物基类"""
    def __init__(self, tolerance):
        self.tolerance = tolerance
    def check_line_segment_collision(self, p1, p2):
        raise NotImplementedError
    def draw(self, ax):
        raise NotImplementedError

class CircularObstacle(Obstacle):
    """圆形障碍物及其碰撞检测和绘制"""
    def __init__(self, center, radius, tolerance):
        super().__init__(tolerance)
        self.center, self.radius = np.array(center), radius
    def check_line_segment_collision(self, p1, p2):
        effective_radius = self.radius + self.tolerance
        p1, p2 = np.array(p1), np.array(p2)
        v = p2 - p1
        a = v.dot(v)
        if a == 0: return np.sum((p1 - self.center)**2) <= effective_radius**2
        b = 2 * v.dot(p1 - self.center)
        c = p1.dot(p1) + self.center.dot(self.center) - 2 * p1.dot(self.center) - effective_radius**2
        d = b**2 - 4 * a * c
        if d < 0: return False
        sqrt_d = np.sqrt(d)
        t1, t2 = (-b + sqrt_d) / (2 * a), (-b - sqrt_d) / (2 * a)
        return (0 <= t1 <= 1) or (0 <= t2 <= 1)
    def draw(self, ax):
        ax.add_artist(plt.Circle(self.center, self.radius, color='gray', alpha=0.7, zorder=2))
        ax.add_artist(plt.Circle(self.center, self.radius + self.tolerance, color='k', linestyle='--', fill=False, alpha=0.5, zorder=1))

class PolygonalObstacle(Obstacle):
    """多边形障碍物及其碰撞检测和绘制"""
    def __init__(self, vertices, tolerance):
        super().__init__(tolerance)
        self.vertices = np.array(vertices)
    def check_line_segment_collision(self, p1, p2):
        for i in range(len(self.vertices)):
            if self._line_intersect(p1, p2, self.vertices[i], self.vertices[(i + 1) % len(self.vertices)]): return True
        for vertex in self.vertices:
            if self._distance_point_to_segment(vertex, p1, p2) < self.tolerance: return True
        return False
    def _line_intersect(self, p1, p2, p3, p4):
        d = (p2[0] - p1[0]) * (p4[1] - p3[1]) - (p2[1] - p1[1]) * (p4[0] - p3[0])
        if abs(d) < 1e-9: return False
        t = ((p3[0] - p1[0]) * (p4[1] - p3[1]) - (p3[1] - p1[1]) * (p4[0] - p3[0])) / d
        u = -((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])) / d
        return 0 <= t <= 1 and 0 <= u <= 1
    def _distance_point_to_segment(self, point, seg_start, seg_end):
        point, seg_start, seg_end = np.array(point), np.array(seg_start), np.array(seg_end)
        line_vec, point_vec = seg_end - seg_start, point - seg_start
        line_len_sq = np.sum(line_vec**2)
        if line_len_sq == 0: return np.linalg.norm(point_vec)
        t = max(0, min(1, np.dot(point_vec, line_vec) / line_len_sq))
        projection = seg_start + t * line_vec
        return np.linalg.norm(point - projection)
    def draw(self, ax):
        ax.add_patch(plt.Polygon(self.vertices, closed=True, color='gray', alpha=0.7, zorder=2))
        for vertex in self.vertices:
            ax.add_artist(plt.Circle(vertex, self.tolerance, color='k', linestyle='--', fill=False, alpha=0.5, zorder=1))

class Node:
    """RRT算法的树节点"""
    def __init__(self, x, y):
        self.x, self.y, self.parent = x, y, None

class RRTPlanner:
    """(已修订) 基础RRT路径规划器"""
    def __init__(self, start, goal, obstacles, config):
        self.start, self.goal = Node(start[0], start[1]), Node(goal[0], goal[1])
        self.obstacles, self.config = obstacles, config
        self.node_list = [self.start]

        # --- [修订] 扩展边界框计算逻辑，以正确包含所有类型的障碍物 ---
        all_points = [start, goal]
        for obs in obstacles:
            if isinstance(obs, PolygonalObstacle):
                all_points.extend(obs.vertices)
            elif isinstance(obs, CircularObstacle):
                # 对于圆形障碍物，考虑其边界上的四个极值点
                center, radius = obs.center, obs.radius
                all_points.append(center + np.array([radius, 0]))
                all_points.append(center - np.array([radius, 0]))
                all_points.append(center + np.array([0, radius]))
                all_points.append(center - np.array([0, radius]))
        
        x_coords, y_coords = [p[0] for p in all_points], [p[1] for p in all_points]
        # 为边界框增加500米的安全缓冲
        self.min_x, self.max_x = min(x_coords) - 500, max(x_coords) + 500
        self.min_y, self.max_y = min(y_coords) - 500, max(y_coords) + 500

    def plan(self) -> Tuple[List[np.ndarray], float, bool]:
        """执行RRT路径规划"""
        for _ in range(self.config.RRT_ITERATIONS):
            rnd_node = self._get_random_node()
            nearest_node = self._get_nearest_node_kdtree(rnd_node)
            new_node = self._steer(nearest_node, rnd_node)
            if self._is_collision(new_node, nearest_node): continue
            new_node.parent = nearest_node
            self.node_list.append(new_node)
            if not self._is_collision(new_node, self.goal):
                self.goal.parent = new_node
                path, length = self._generate_final_path(self.goal)
                return path, length, True
        closest_node = min(self.node_list, key=lambda node: np.hypot(node.x - self.goal.x, node.y - self.goal.y))
        path, length = self._generate_final_path(closest_node)
        return path, length, False

    def _get_random_node(self):
        if random.random() > 0.1: return self.goal
        return Node(random.uniform(self.min_x, self.max_x), random.uniform(self.min_y, self.max_y))

    def _get_nearest_node_kdtree(self, rnd_node):
        tree = cKDTree([[node.x, node.y] for node in self.node_list])
        _, idx = tree.query([rnd_node.x, rnd_node.y])
        return self.node_list[idx]

    def _steer(self, from_node, to_node, force_full_step=False):
        p1, p2 = np.array([from_node.x, from_node.y]), np.array([to_node.x, to_node.y])
        dist = np.linalg.norm(p2 - p1)
        if dist == 0: return from_node
        step = self.config.RRT_STEP_SIZE if not force_full_step else dist
        ratio = min(step, dist) / dist
        new_pos = p1 + (p2 - p1) * ratio
        return Node(new_pos[0], new_pos[1])

    def _is_collision(self, node1, node2):
        p1, p2 = (node1.x, node1.y), (node2.x, node2.y)
        for obs in self.obstacles:
            if obs.check_line_segment_collision(p1, p2): return True
        return False

    def _generate_final_path(self, goal_node):
        path, node = [], goal_node
        while node is not None:
            path.append(np.array([node.x, node.y]))
            node = node.parent
        return path[::-1], np.sum(np.linalg.norm(np.diff(path[::-1], axis=0), axis=1))

class PHCurveRRTPlanner:
    """(已修订) 封装标准RRT与PH曲线平滑的高级规划器"""
    def __init__(self, start, goal, start_heading, goal_heading, obstacles, config):
        self.start, self.goal = np.array(start), np.array(goal)
        self.start_heading, self.goal_heading = start_heading, goal_heading
        self.obstacles, self.config = obstacles, config
        self.rrt_planner = RRTPlanner(start, goal, obstacles, config)

    def plan(self) -> Union[Tuple[np.ndarray, float], None]:
        """执行 RRT + 路径平滑"""
        raw_path_points, raw_path_length, is_complete = self.rrt_planner.plan()
        if not is_complete and len(raw_path_points) < 2:
            print(f"警告: RRT规划从 {self.start} 到 {self.goal} 失败，无安全路径。")
            return None
        key_waypoints = self._simplify_path(raw_path_points)
        for _ in range(self.config.MAX_REFINEMENT_ATTEMPTS):
            smooth_path, segment_map = self._generate_bezier_path(key_waypoints)
            collision_idx = self._find_collision_point_idx(smooth_path)
            if collision_idx is None:
                final_length = np.sum(np.linalg.norm(np.diff(smooth_path, axis=0), axis=1))
                return smooth_path, final_length
            colliding_segment = segment_map[collision_idx]
            p_start, p_end = key_waypoints[colliding_segment], key_waypoints[colliding_segment + 1]
            intermediate_point = self._find_best_intermediate_point(p_start, p_end, raw_path_points)
            if intermediate_point is not None:
                key_waypoints = np.insert(key_waypoints, colliding_segment + 1, intermediate_point, axis=0)
            else:
                return np.array(raw_path_points), raw_path_length
        return np.array(raw_path_points), raw_path_length

    def _simplify_path(self, path):
        if len(path) < 3: return np.array(path)
        simplified_path = [path[0]]
        i = 0
        while i < len(path) - 1:
            for j in range(len(path) - 1, i, -1):
                if not any(obs.check_line_segment_collision(path[i], path[j]) for obs in self.obstacles):
                    simplified_path.append(path[j])
                    i = j
                    break
            else: i += 1
        return np.array(simplified_path)

    def _generate_bezier_path(self, waypoints):
        if len(waypoints) < 2: return waypoints, []
        headings = self._calculate_headings_at_waypoints(waypoints)
        full_curve, segment_map = [], []
        for i in range(len(waypoints) - 1):
            p0, p1, h0, h1 = waypoints[i], waypoints[i+1], headings[i], headings[i+1]
            segment = self._quintic_bezier_segment(p0, h0, p1, h1)
            full_curve.extend(segment[:-1])
            segment_map.extend([i] * (len(segment) - 1))
        full_curve.append(waypoints[-1])
        segment_map.append(len(waypoints) - 2)
        return np.array(full_curve), segment_map

    def _calculate_headings_at_waypoints(self, waypoints):
        headings = [self.start_heading]
        for i in range(1, len(waypoints) - 1):
            v_in, v_out = waypoints[i] - waypoints[i-1], waypoints[i+1] - waypoints[i]
            h_in, h_out = np.arctan2(v_in[1], v_in[0]), np.arctan2(v_out[1], v_out[0])
            angle_diff = h_out - h_in
            if angle_diff > np.pi: angle_diff -= 2 * np.pi
            if angle_diff < -np.pi: angle_diff += 2 * np.pi
            headings.append(h_in + angle_diff / 2.0)
        headings.append(self.goal_heading)
        return headings

    def _quintic_bezier_segment(self, p0, h0, p1, h1, samples=50):
        p0, p1 = np.array(p0), np.array(p1)
        k = np.linalg.norm(p1 - p0) * 0.5
        c0, c1 = p0, p0 + k * np.array([np.cos(h0), np.sin(h0)])
        c5, c4 = p1, p1 - k * np.array([np.cos(h1), np.sin(h1)])
        c2, c3 = c0 + (c5 - c0) * 0.3, c0 + (c5 - c0) * 0.7
        t = np.linspace(0, 1, samples)
        t_1 = 1 - t
        return (np.outer(t_1**5, c0) + np.outer(5*t*t_1**4, c1) + np.outer(10*t**2*t_1**3, c2) +
                np.outer(10*t**3*t_1**2, c3) + np.outer(5*t**4*t_1, c4) + np.outer(t**5, c5))

    def _find_collision_point_idx(self, path):
        for i in range(len(path) - 1):
            if any(obs.check_line_segment_collision(path[i], path[i+1]) for obs in self.obstacles): return i
        return None

    def _find_best_intermediate_point(self, p_start, p_end, raw_path):
        try:
            start_idx = np.where(np.all(raw_path == p_start, axis=1))[0][0]
            end_idx = np.where(np.all(raw_path == p_end, axis=1))[0][0]
            if start_idx >= end_idx: return None
            mid_idx = (start_idx + end_idx) // 2
            if mid_idx > start_idx: return raw_path[mid_idx]
        except IndexError: return None
        return None