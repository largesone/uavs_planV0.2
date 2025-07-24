# -*- coding: utf-8 -*-
# 文件名: networks.py
# 描述: 统一的神经网络模块，包含所有网络结构定义

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class SimpleNetwork(nn.Module):
    """简化的网络结构 - 基础版本"""
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout: float = 0.1):
        super(SimpleNetwork, self).__init__()
        
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

class DeepFCN(nn.Module):
    """深度全连接网络"""
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout: float = 0.1):
        super(DeepFCN, self).__init__()
        
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

class GATNetwork(nn.Module):
    """图注意力网络 - 简化版本"""
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout: float = 0.1):
        super(GATNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        
        # 嵌入维度
        self.embedding_dim = 128
        
        # 实体特征维度 - 确保维度合理
        entity_feature_dim = max(input_dim // 2, 32)  # 最小32维
        
        # 实体编码器 - 简化版本避免维度问题
        self.uav_encoder = nn.Sequential(
            nn.Linear(entity_feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.target_encoder = nn.Sequential(
            nn.Linear(entity_feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # 输出层 - 简化版本
        self.output_layer = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.3)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _extract_entity_features(self, state):
        """从状态向量中提取实体特征"""
        batch_size = state.shape[0]
        feature_dim = state.shape[1]
        
        # 确保特征维度合理
        if feature_dim < 64:
            # 如果状态维度太小，使用填充
            padded_state = torch.zeros(batch_size, 64, device=state.device)
            padded_state[:, :feature_dim] = state
            state = padded_state
            feature_dim = 64
        
        split_point = feature_dim // 2
        
        uav_features = state[:, :split_point]
        target_features = state[:, split_point:]
        
        return uav_features, target_features
    
    def forward(self, x):
        """前向传播"""
        # 处理BatchNorm在单样本时的问题
        if x.shape[0] == 1:
            training_mode = self.training
            self.eval()
        else:
            training_mode = None
        
        try:
            # 提取实体特征
            uav_features, target_features = self._extract_entity_features(x)
            
            # 编码实体
            uav_embedding = self.uav_encoder(uav_features)
            target_embedding = self.target_encoder(target_features)
            
            # 注意力机制
            combined_embedding = torch.cat([uav_embedding, target_embedding], dim=1)
            
            # 输出层
            output = self.output_layer(combined_embedding)
            
            return output
        
        finally:
            if training_mode is not None:
                self.train(training_mode)

class DeepFCNResidual(nn.Module):
    """带残差连接的深度全连接网络 - 优化版本"""
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout: float = 0.2):
        super(DeepFCNResidual, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims if hidden_dims else [256, 128, 64]  # 默认层次结构
        self.output_dim = output_dim
        self.dropout = dropout
        
        # 输入层 - 添加BatchNorm
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dims[0]),
            nn.BatchNorm1d(self.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)  # 输入层使用较小的dropout
        )
        
        # 残差块
        self.residual_blocks = nn.ModuleList()
        for i in range(len(self.hidden_dims) - 1):
            block = ResidualBlock(self.hidden_dims[i], self.hidden_dims[i+1], dropout)
            self.residual_blocks.append(block)
        
        # 注意力机制 - 简化版本
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1] // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[-1] // 4, self.hidden_dims[-1]),
            nn.Sigmoid()
        )
        
        # 输出层 - 优化结构
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, output_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """改进的权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 使用He初始化，适合ReLU激活函数
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)  # 小的正偏置
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """前向传播 - 添加注意力机制"""
        # 输入层
        x = self.input_layer(x)
        
        # 残差块
        for block in self.residual_blocks:
            x = block(x)
        
        # 注意力机制
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        # 输出层
        return self.output_layer(x)

class ResidualBlock(nn.Module):
    """优化的残差块 - 改进的结构和正则化"""
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super(ResidualBlock, self).__init__()
        
        # 主路径 - 使用预激活结构
        self.layers = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        
        # 跳跃连接
        if in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim)
            )
        else:
            self.shortcut = nn.Identity()
        
        # 添加SE注意力模块
        self.se_attention = SEBlock(out_dim, reduction=4)
    
    def forward(self, x):
        """前向传播 - 预激活残差连接"""
        residual = self.shortcut(x)
        out = self.layers(x)
        
        # 应用SE注意力
        out = self.se_attention(out)
        
        # 残差连接
        return out + residual

class SEBlock(nn.Module):
    """Squeeze-and-Excitation注意力块"""
    def __init__(self, channels: int, reduction: int = 16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """前向传播"""
        # 全局平均池化
        b, c = x.size()
        y = x.mean(dim=0, keepdim=True)  # 简化的全局池化
        
        # 激励操作
        y = self.excitation(y)
        
        # 重新加权
        return x * y.expand_as(x)

def create_network(network_type: str, input_dim: int, hidden_dims: List[int], output_dim: int) -> nn.Module:
    """
    创建指定类型的网络
    
    Args:
        network_type: 网络类型 ("SimpleNetwork", "DeepFCN", "GAT", "DeepFCNResidual")
        input_dim: 输入维度
        hidden_dims: 隐藏层维度列表
        output_dim: 输出维度
        
    Returns:
        网络模型实例,SimpleNetwork、DeepFCN、GAT、DeepFCNResidual
    """
    if network_type == "SimpleNetwork":
        return SimpleNetwork(input_dim, hidden_dims, output_dim)
    elif network_type == "DeepFCN":
        return DeepFCN(input_dim, hidden_dims, output_dim)
    elif network_type == "GAT":
        return GATNetwork(input_dim, hidden_dims, output_dim)
    elif network_type == "DeepFCNResidual":
        return DeepFCNResidual(input_dim, hidden_dims, output_dim)
    else:
        raise ValueError(f"不支持的网络类型: {network_type}")

def get_network_info(network_type: str) -> dict:
    """
    获取网络信息
    
    Args:
        network_type: 网络类型
        
    Returns:
        网络信息字典
    """
    network_info = {
        "SimpleNetwork": {
            "description": "基础全连接网络",
            "features": ["BatchNorm", "Dropout", "Xavier初始化"],
            "complexity": "低"
        },
        "DeepFCN": {
            "description": "深度全连接网络",
            "features": ["多层结构", "BatchNorm", "Dropout"],
            "complexity": "中"
        },
        "GAT": {
            "description": "图注意力网络",
            "features": ["注意力机制", "实体编码", "多模态融合"],
            "complexity": "高"
        },
        "DeepFCNResidual": {
            "description": "带残差连接的深度网络",
            "features": ["残差连接", "BatchNorm", "Dropout"],
            "complexity": "中"
        }
    }
    
    return network_info.get(network_type, {"description": "未知网络", "features": [], "complexity": "未知"}) 