#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAY库网络结构迁移版本
包含GAT网络和深度残差网络，用于对比不同的结构影响
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional

class GATNetwork(nn.Module):
    """图注意力网络 (Graph Attention Network) - RAY库版本"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, 
                 num_heads: int = 8, dropout: float = 0.1):
        super(GATNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        # 嵌入维度
        self.embedding_dim = 128
        
        # 实体特征维度
        entity_feature_dim = input_dim // 2
        
        # 1. 实体编码器
        self.uav_encoder = nn.Sequential(
            nn.Linear(entity_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(128),
            nn.Linear(128, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(self.embedding_dim)
        )
        
        self.target_encoder = nn.Sequential(
            nn.Linear(entity_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(128),
            nn.Linear(128, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(self.embedding_dim)
        )
        
        # 2. GAT层
        self.gat_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.embedding_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            ) for _ in range(3)  # 3层GAT
        ])
        
        # 3. 全局上下文编码器
        self.global_context_encoder = nn.Sequential(
            nn.Linear(self.embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(32)
        )
        
        # 4. 动作解码器
        policy_input_dim = self.embedding_dim * 2 + 32
        self.action_decoder = nn.Sequential(
            nn.Linear(policy_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1)
        )
        
        # 残差连接
        self.uav_residual = nn.Linear(entity_feature_dim, self.embedding_dim)
        self.target_residual = nn.Linear(entity_feature_dim, self.embedding_dim)
        self.action_residual = nn.Linear(policy_input_dim, 1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.3)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _extract_entity_features(self, state):
        """提取实体特征"""
        batch_size = state.shape[0]
        feature_dim = state.shape[1]
        
        # 动态分割状态向量
        split_point = feature_dim // 2
        
        uav_features = state[:, :split_point]
        target_features = state[:, split_point:]
        
        return uav_features, target_features
    
    def forward(self, x):
        """前向传播 - GAT网络"""
        batch_size = x.shape[0]
        
        # 处理BatchNorm在单样本时的问题
        if batch_size == 1:
            training_mode = self.training
            self.eval()
        else:
            training_mode = None
        
        try:
            # 1. 实体编码
            uav_features, target_features = self._extract_entity_features(x)
            
            # UAV编码 - 带残差连接
            uav_embedding_main = self.uav_encoder(uav_features)
            uav_embedding_residual = self.uav_residual(uav_features)
            uav_embedding = uav_embedding_main + uav_embedding_residual
            
            # 目标编码 - 带残差连接
            target_embedding_main = self.target_encoder(target_features)
            target_embedding_residual = self.target_residual(target_features)
            target_embedding = target_embedding_main + target_embedding_residual
            
            # 2. GAT层处理
            # 扩展为序列形式
            uav_embedding = uav_embedding.unsqueeze(1)  # [batch, 1, embedding_dim]
            target_embedding = target_embedding.unsqueeze(1)  # [batch, 1, embedding_dim]
            
            # 合并所有实体嵌入
            all_embeddings = torch.cat([uav_embedding, target_embedding], dim=1)
            
            # 通过GAT层
            gat_output = all_embeddings
            for gat_layer in self.gat_layers:
                gat_output, _ = gat_layer(gat_output, gat_output, gat_output)
            
            # 3. 注意力聚合
            uav_gat_output = gat_output[:, 0:1]  # UAV的GAT输出
            target_gat_output = gat_output[:, 1:]  # 目标的GAT输出
            
            # 使用UAV作为query，目标作为key和value
            attended_output, _ = self.gat_layers[0](
                query=uav_gat_output,
                key=target_gat_output,
                value=target_gat_output
            )
            
            context_vector = attended_output.squeeze(1)
            
            # 4. 全局上下文
            global_context = torch.mean(target_gat_output, dim=1)
            g_context = self.global_context_encoder(global_context)
            
            # 5. 动作解码
            policy_input = torch.cat([
                uav_gat_output.squeeze(1),
                context_vector,
                g_context
            ], dim=1)
            
            # 检查维度匹配
            expected_dim = self.embedding_dim * 2 + 32
            actual_dim = policy_input.shape[1]
            
            if actual_dim != expected_dim:
                # 动态调整
                if not hasattr(self, '_action_decoder_fixed'):
                    self._action_decoder_fixed = nn.Sequential(
                        nn.Linear(actual_dim, 256),
                        nn.ReLU(),
                        nn.Dropout(self.dropout),
                        nn.BatchNorm1d(256),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Dropout(self.dropout),
                        nn.BatchNorm1d(128),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Dropout(self.dropout),
                        nn.BatchNorm1d(64),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Dropout(self.dropout),
                        nn.BatchNorm1d(32),
                        nn.Linear(32, 1)
                    )
                    self._action_residual_fixed = nn.Linear(actual_dim, 1)
                
                action_decoder = self._action_decoder_fixed
                action_residual = self._action_residual_fixed
            else:
                action_decoder = self.action_decoder
                action_residual = self.action_residual
            
            # 解码Q值 - 带残差连接
            q_value_main = action_decoder(policy_input)
            q_value_residual = action_residual(policy_input)
            q_value = q_value_main + q_value_residual
            
            # 扩展到完整的动作空间
            if batch_size == 1:
                expanded_q_values = q_value.expand(1, self.output_dim)
            else:
                expanded_q_values = q_value.expand(batch_size, self.output_dim)
            
            return expanded_q_values
            
        finally:
            # 恢复训练模式
            if training_mode is not None:
                if training_mode:
                    self.train()
                else:
                    self.eval()


class DeepResidualNetwork(nn.Module):
    """深度残差网络 - RAY库版本"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, 
                 dropout: float = 0.1):
        super(DeepResidualNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        
        # 构建残差块
        self.layers = nn.ModuleList()
        self.residual_projections = nn.ModuleList()
        
        # 输入层
        current_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            # 主路径
            layer = nn.Sequential(
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim)
            )
            self.layers.append(layer)
            
            # 残差投影层
            if i == 0:
                residual_proj = nn.Linear(input_dim, hidden_dim)
            else:
                residual_proj = nn.Linear(hidden_dims[i-1], hidden_dim)
            self.residual_projections.append(residual_proj)
            
            current_dim = hidden_dim
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(current_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(32),
            nn.Linear(32, output_dim)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.3)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """前向传播 - 深度残差网络"""
        residual_input = x
        
        # 通过残差层
        for i, (layer, residual_proj) in enumerate(zip(self.layers, self.residual_projections)):
            # 主路径
            main_output = layer(x)
            
            # 残差路径
            if i == 0:
                residual_output = residual_proj(residual_input)
            else:
                residual_output = residual_proj(x)
            
            # 残差连接
            x = main_output + residual_output
        
        # 输出层
        output = self.output_layer(x)
        
        return output


class SimpleFCN(nn.Module):
    """简单全连接网络 - 作为基线对比"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, 
                 dropout: float = 0.1):
        super(SimpleFCN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        
        # 构建网络层
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            current_dim = hidden_dim
        
        # 输出层
        layers.extend([
            nn.Linear(current_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(64),
            nn.Linear(64, output_dim)
        ])
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.3)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """前向传播"""
        return self.network(x)


def create_network(network_type: str, input_dim: int, hidden_dims: List[int], 
                  output_dim: int, dropout: float = 0.1) -> nn.Module:
    """创建指定类型的网络"""
    
    if network_type == 'GAT':
        return GATNetwork(input_dim, hidden_dims, output_dim, dropout=dropout)
    elif network_type == 'DeepResidual':
        return DeepResidualNetwork(input_dim, hidden_dims, output_dim, dropout=dropout)
    elif network_type == 'SimpleFCN':
        return SimpleFCN(input_dim, hidden_dims, output_dim, dropout=dropout)
    else:
        raise ValueError(f"未知的网络类型: {network_type}")


def get_network_info(network: nn.Module) -> dict:
    """获取网络信息"""
    total_params = sum(p.numel() for p in network.parameters())
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'network_type': network.__class__.__name__
    } 