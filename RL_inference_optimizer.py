# -*- coding: utf-8 -*-
# 文件名: RL_inference_optimizer.py
# 描述: RL推理优化器，采用多次推理方式，取最优结果输出

import numpy as np
import time
import torch
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import random

from entities import UAV, Target
from main import GraphRLSolver, DirectedGraph
from config import Config

class RLInferenceOptimizer:
    """RL推理优化器 - 多次推理取最优"""
    
    def __init__(self, uavs: List[UAV], targets: List[Target], obstacles, config: Config):
        self.uavs = uavs
        self.targets = targets
        self.obstacles = obstacles
        self.config = config
        
        # 推理参数
        self.n_inference_runs = getattr(config, 'RL_N_INFERENCE_RUNS', 10)
        self.inference_temperature = getattr(config, 'RL_INFERENCE_TEMPERATURE', 0.1)
        self.ensemble_size = getattr(config, 'RL_ENSEMBLE_SIZE', 3)
        
        # 构建图结构
        self.graph = DirectedGraph(uavs, targets, config.GRAPH_N_PHI, obstacles)
        
        # 初始化RL求解器
        self.rl_solvers = []
        self._initialize_rl_solvers()
        
        print(f"RL推理优化器已初始化 (推理次数: {self.n_inference_runs}, 集成大小: {self.ensemble_size})")
    
    def _initialize_rl_solvers(self):
        """初始化多个RL求解器"""
        for i in range(self.ensemble_size):
            try:
                # 为每个求解器设置不同的随机种子
                random.seed(42 + i)
                np.random.seed(42 + i)
                torch.manual_seed(42 + i)
                
                # 创建RL求解器
                solver = GraphRLSolver(
                    self.uavs, self.targets, self.graph, self.obstacles,
                    i_dim=len(self.uavs) * 3 + len(self.targets) * 3,
                    h_dim=128,
                    o_dim=len(self.uavs) * len(self.targets) * self.graph.n_phi,
                    config=self.config
                )
                
                # 尝试加载预训练模型
                try:
                    if hasattr(self.config, 'SAVED_MODEL_PATH') and self.config.SAVED_MODEL_PATH:
                        solver.load_model(self.config.SAVED_MODEL_PATH)
                        print(f"✓ RL求解器 {i+1} 加载预训练模型成功")
                    else:
                        print(f"⚠ RL求解器 {i+1} 使用随机初始化")
                except Exception as e:
                    print(f"⚠ RL求解器 {i+1} 模型加载失败: {e}")
                
                self.rl_solvers.append(solver)
                
            except Exception as e:
                print(f"✗ RL求解器 {i+1} 初始化失败: {e}")
    
    def _run_single_inference(self, solver: GraphRLSolver, temperature: float = 0.1) -> Tuple[Dict, float]:
        """运行单次推理"""
        try:
            # 设置推理温度
            original_epsilon = solver.epsilon
            solver.epsilon = temperature
            
            # 运行推理
            start_time = time.time()
            plan, training_time, planning_time = solver.solve()
            inference_time = time.time() - start_time
            
            # 恢复原始探索率
            solver.epsilon = original_epsilon
            
            return plan, inference_time
            
        except Exception as e:
            print(f"单次推理失败: {e}")
            return {}, 0.0
    
    def _evaluate_plan_quality(self, plan: Dict) -> float:
        """评估方案质量"""
        if not plan:
            return -1000
        
        total_reward = 0
        total_assignments = 0
        total_distance = 0
        
        for uav_id, uav_plan in plan.items():
            if 'targets' in uav_plan:
                assignments = uav_plan['targets']
                total_assignments += len(assignments)
                
                uav = next(u for u in self.uavs if u.id == uav_id)
                uav_distance = 0
                
                for target_id, phi_idx in assignments:
                    target = next(t for t in self.targets if t.id == target_id)
                    
                    # 计算距离
                    distance = np.linalg.norm(np.array(uav.position) - np.array(target.position))
                    uav_distance += distance
                    total_distance += distance
                    
                    # 计算奖励
                    target_value = target.value
                    distance_penalty = distance * 0.01
                    reward = target_value - distance_penalty
                    total_reward += reward
        
        # 完成率奖励
        completion_rate = total_assignments / len(self.targets) if len(self.targets) > 0 else 0
        completion_bonus = completion_rate * 500
        
        # 负载均衡奖励
        assignment_counts = [len(plan.get(uav.id, {}).get('targets', [])) for uav in self.uavs]
        if len(assignment_counts) > 1:
            load_balance = 1.0 - np.std(assignment_counts) / (np.mean(assignment_counts) + 1e-6)
            balance_bonus = load_balance * 200
        else:
            balance_bonus = 0
        
        quality_score = total_reward + completion_bonus + balance_bonus
        
        return quality_score
    
    def _ensemble_inference(self) -> List[Tuple[Dict, float, float]]:
        """集成推理 - 使用多个模型"""
        ensemble_results = []
        
        for i, solver in enumerate(self.rl_solvers):
            print(f"运行集成推理 {i+1}/{len(self.rl_solvers)}...")
            
            # 多次推理
            for run in range(self.n_inference_runs):
                plan, inference_time = self._run_single_inference(solver, self.inference_temperature)
                quality_score = self._evaluate_plan_quality(plan)
                
                ensemble_results.append((plan, quality_score, inference_time))
                
                if run % 5 == 0:
                    print(f"  推理 {run+1}/{self.n_inference_runs}, 质量评分: {quality_score:.2f}")
        
        return ensemble_results
    
    def _select_best_plan(self, ensemble_results: List[Tuple[Dict, float, float]]) -> Tuple[Dict, float, float]:
        """选择最优方案"""
        if not ensemble_results:
            return {}, -1000, 0.0
        
        # 按质量评分排序
        sorted_results = sorted(ensemble_results, key=lambda x: x[1], reverse=True)
        
        best_plan, best_score, best_time = sorted_results[0]
        
        print(f"最优方案质量评分: {best_score:.2f}")
        print(f"平均质量评分: {np.mean([r[1] for r in ensemble_results]):.2f}")
        print(f"质量评分标准差: {np.std([r[1] for r in ensemble_results]):.2f}")
        
        return best_plan, best_score, best_time
    
    def optimize_inference(self) -> Tuple[Dict, float, float]:
        """优化推理过程"""
        print("开始RL推理优化...")
        
        start_time = time.time()
        
        # 1. 集成推理
        ensemble_results = self._ensemble_inference()
        
        # 2. 选择最优方案
        best_plan, best_score, best_inference_time = self._select_best_plan(ensemble_results)
        
        # 3. 后处理优化
        optimized_plan = self._post_process_plan(best_plan)
        
        total_time = time.time() - start_time
        
        print(f"RL推理优化完成，总耗时: {total_time:.2f}s")
        print(f"最优方案质量评分: {best_score:.2f}")
        
        return optimized_plan, total_time, best_inference_time
    
    def _post_process_plan(self, plan: Dict) -> Dict:
        """后处理优化方案"""
        if not plan:
            return plan
        
        # 1. 移除无效分配
        cleaned_plan = {}
        for uav_id, uav_plan in plan.items():
            if 'targets' in uav_plan and uav_plan['targets']:
                # 检查每个分配的有效性
                valid_assignments = []
                for target_id, phi_idx in uav_plan['targets']:
                    uav = next(u for u in self.uavs if u.id == uav_id)
                    target = next(t for t in self.targets if t.id == target_id)
                    
                    # 检查约束
                    distance = np.linalg.norm(np.array(uav.position) - np.array(target.position))
                    if distance <= uav.max_distance * 1.2:  # 允许20%超出
                        resource_match = np.all(uav.resources >= target.resources * 0.8)
                        if resource_match:
                            valid_assignments.append((target_id, phi_idx))
                
                if valid_assignments:
                    cleaned_plan[uav_id] = {
                        'targets': valid_assignments,
                        'path': uav_plan.get('path', []),
                        'total_distance': uav_plan.get('total_distance', 0.0),
                        'completion_time': uav_plan.get('completion_time', 0.0)
                    }
        
        return cleaned_plan
    
    def get_inference_statistics(self) -> Dict:
        """获取推理统计信息"""
        stats = {
            'n_solvers': len(self.rl_solvers),
            'n_inference_runs': self.n_inference_runs,
            'total_inference_runs': len(self.rl_solvers) * self.n_inference_runs,
            'inference_temperature': self.inference_temperature,
            'ensemble_size': self.ensemble_size
        }
        
        return stats

def test_rl_inference_optimizer():
    """测试RL推理优化器"""
    print("=" * 60)
    print("测试RL推理优化器")
    print("=" * 60)
    
    # 获取测试场景
    from scenarios import get_small_scenario
    uavs, targets, obstacles = get_small_scenario(obstacle_tolerance=50.0)
    config = Config()
    
    # 创建优化器
    optimizer = RLInferenceOptimizer(uavs, targets, obstacles, config)
    
    # 运行优化推理
    plan, total_time, inference_time = optimizer.optimize_inference()
    
    # 输出统计信息
    stats = optimizer.get_inference_statistics()
    print(f"\n推理统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 评估结果
    total_assignments = sum(len(plan.get(uav.id, {}).get('targets', [])) for uav in uavs)
    completion_rate = total_assignments / len(targets) if len(targets) > 0 else 0
    
    print(f"\n优化结果:")
    print(f"  完成率: {completion_rate:.2%}")
    print(f"  分配任务数: {total_assignments}/{len(targets)}")
    print(f"  总耗时: {total_time:.2f}s")
    print(f"  推理耗时: {inference_time:.2f}s")
    
    return plan, total_time, completion_rate

if __name__ == "__main__":
    plan, total_time, completion_rate = test_rl_inference_optimizer()
    print(f"\n{'='*60}")
    print("测试完成！")
    print(f"{'='*60}") 