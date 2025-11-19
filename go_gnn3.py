"""
GO Term GNN - 完整实现（修复版v3）
修复了embedding collapse和维度不匹配问题
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import GATConv, GCNConv, HeteroConv
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter, deque
from typing import Dict, List, Tuple, Set, Optional
import pickle
import json
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. OBO文件解析器 ====================
class OBOParser:
    """解析GO的OBO格式文件"""
    
    def __init__(self, obo_file_path: str):
        self.obo_file_path = obo_file_path
        self.go_terms = {}
        self.relationships = []
        self.namespace_terms = {
            'biological_process': set(),
            'molecular_function': set(),
            'cellular_component': set()
        }
        
        if not os.path.exists(obo_file_path):
            raise FileNotFoundError(f"OBO file not found: {obo_file_path}")
    
    def parse(self) -> Tuple[Dict, List, Dict]:
        """解析OBO文件"""
        print(f"正在解析文件: {self.obo_file_path}")
        print(f"文件大小: {os.path.getsize(self.obo_file_path) / 1024 / 1024:.2f} MB")
        
        with open(self.obo_file_path, 'r', encoding='utf-8') as f:
            current_term = {}
            in_term = False
            line_count = 0
            
            for line in f:
                line_count += 1
                line = line.strip()
                
                if line == "[Term]":
                    if current_term:
                        self._process_term(current_term)
                    current_term = {}
                    in_term = True
                    
                elif line == "[Typedef]" or (line.startswith("[") and line != "[Term]"):
                    if current_term:
                        self._process_term(current_term)
                    current_term = {}
                    in_term = False
                    
                elif in_term and line:
                    self._parse_term_line(line, current_term)
            
            if current_term:
                self._process_term(current_term)
        
        print(f"✓ 解析完成: {len(self.go_terms):,} 个GO terms, {len(self.relationships):,} 条关系")
        return self.go_terms, self.relationships, self.namespace_terms
    
    def _parse_term_line(self, line: str, current_term: dict):
        """解析term块中的一行"""
        if line.startswith("id:"):
            current_term['id'] = line.split("id:")[1].strip()
        elif line.startswith("name:"):
            current_term['name'] = line.split("name:")[1].strip()
        elif line.startswith("namespace:"):
            current_term['namespace'] = line.split("namespace:")[1].strip()
        elif line.startswith("def:"):
            def_text = line.split("def:")[1].strip()
            if def_text.startswith('"'):
                def_text = def_text[1:def_text.find('"', 1)]
            current_term['def'] = def_text
        elif line.startswith("is_a:"):
            parent = line.split("is_a:")[1].split("!")[0].strip()
            if 'is_a' not in current_term:
                current_term['is_a'] = []
            current_term['is_a'].append(parent)
        elif line.startswith("relationship:"):
            parts = line.split("relationship:")[1].split("!")
            rel_info = parts[0].strip().split()
            if len(rel_info) >= 2:
                rel_type = rel_info[0]
                target = rel_info[1]
                if 'relationship' not in current_term:
                    current_term['relationship'] = []
                current_term['relationship'].append((rel_type, target))
        elif line.startswith("is_obsolete:"):
            if "true" in line.lower():
                current_term['is_obsolete'] = True
    
    def _process_term(self, term: dict):
        """处理并存储一个GO term"""
        if 'id' not in term or term.get('is_obsolete', False):
            return
        
        go_id = term['id']
        self.go_terms[go_id] = term
        
        if 'namespace' in term:
            ns = term['namespace']
            if ns in self.namespace_terms:
                self.namespace_terms[ns].add(go_id)
        
        if 'is_a' in term:
            for parent in term['is_a']:
                self.relationships.append((go_id, parent, 'is_a'))
        
        if 'relationship' in term:
            for rel_type, target in term['relationship']:
                self.relationships.append((go_id, target, rel_type))

# ==================== 2. 层次结构分析器 ====================
class HierarchyAnalyzer:
    """分析GO图的层次结构"""
    
    def __init__(self, go_terms, relationships):
        self.go_terms = go_terms
        self.relationships = relationships
        self.graph = nx.DiGraph()
        self._build_graph()
        
    def _build_graph(self):
        """构建NetworkX图"""
        for source, target, rel_type in self.relationships:
            if source in self.go_terms and target in self.go_terms:
                self.graph.add_edge(source, target, relation=rel_type)
    
    def compute_depth(self, go_id: str) -> int:
        """计算GO term的深度"""
        roots = [n for n in self.graph.nodes() if self.graph.out_degree(n) == 0]
        
        if not roots:
            return 0
        
        min_depth = float('inf')
        for root in roots:
            try:
                if nx.has_path(self.graph, go_id, root):
                    depth = nx.shortest_path_length(self.graph, go_id, root)
                    min_depth = min(min_depth, depth)
            except nx.NetworkXError:
                continue
        
        return min_depth if min_depth != float('inf') else 0
    
    def compute_all_depths(self) -> Dict[str, int]:
        """计算所有节点的深度"""
        depths = {}
        roots = [n for n in self.graph.nodes() if self.graph.out_degree(n) == 0]
        
        print("  计算节点深度...")
        for go_id in tqdm(self.go_terms.keys(), desc="  深度计算"):
            min_depth = float('inf')
            for root in roots:
                try:
                    if nx.has_path(self.graph, go_id, root):
                        depth = nx.shortest_path_length(self.graph, go_id, root)
                        min_depth = min(min_depth, depth)
                except:
                    continue
            depths[go_id] = min_depth if min_depth != float('inf') else 0
        
        return depths

# ==================== 3. 层次感知的图构建器 ====================
class HierarchicalGOGraphBuilder:
    """构建层次感知的异构图"""
    
    def __init__(self, go_terms, relationships, namespace_terms):
        self.go_terms = go_terms
        self.relationships = relationships
        self.namespace_terms = namespace_terms
        self.go_id_to_idx = {}
        self.idx_to_go_id = {}
        self.analyzer = HierarchyAnalyzer(go_terms, relationships)
        
    def build_hierarchical_graph(self):
        """构建层次感知的异构图"""
        print("\n构建层次感知的异构图...")
        
        # 创建节点索引
        for idx, go_id in enumerate(sorted(self.go_terms.keys())):
            self.go_id_to_idx[go_id] = idx
            self.idx_to_go_id[idx] = go_id
        
        num_nodes = len(self.go_terms)
        print(f"  节点数: {num_nodes:,}")
        
        # 构建不同类型的边
        edge_dict = self._build_edge_dict()
        
        # 生成层次感知的节点特征
        node_features = self._generate_hierarchical_features()
        
        # 创建异构图数据
        data = HeteroData()
        data['go_term'].x = node_features
        
        # 添加不同类型的边
        for edge_type, edge_index in edge_dict.items():
            if len(edge_index) > 0:
                edge_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                data['go_term', edge_type, 'go_term'].edge_index = edge_tensor
                print(f"  {edge_type}: {len(edge_index):,} 条边")
        
        # 添加元信息
        data['go_term'].namespace = self._get_namespace_tensor()
        data['go_term'].depth = self._get_depth_tensor()
        data['go_term'].node_degrees = self._get_degree_tensor()
        
        print(f"✓ 图构建完成")
        
        return data, self.go_id_to_idx, self.idx_to_go_id
    
    def _build_edge_dict(self) -> Dict[str, List[List[int]]]:
        """构建不同类型的边"""
        print("  构建边索引...")
        
        edge_dict = {
            'is_a': [],
            'part_of': [],
            'regulates': [],
            'same_level': [],
            'same_namespace': []
        }
        
        # 添加原始关系边
        for source, target, rel_type in self.relationships:
            if source in self.go_id_to_idx and target in self.go_id_to_idx:
                src_idx = self.go_id_to_idx[source]
                tgt_idx = self.go_id_to_idx[target]
                
                if rel_type == 'is_a':
                    edge_dict['is_a'].append([src_idx, tgt_idx])
                elif rel_type == 'part_of':
                    edge_dict['part_of'].append([src_idx, tgt_idx])
                elif 'regulates' in rel_type:
                    edge_dict['regulates'].append([src_idx, tgt_idx])
        
        # 计算深度用于同层连接
        depths = self.analyzer.compute_all_depths()
        
        # 添加同层次边（采样）
        print("  添加同层次连接...")
        level_groups = defaultdict(list)
        for go_id, depth in depths.items():
            if go_id in self.go_id_to_idx:
                level_groups[depth].append(self.go_id_to_idx[go_id])
        
        for level, indices in level_groups.items():
            if len(indices) > 1:
                np.random.shuffle(indices)
                for i in range(len(indices)):
                    for j in range(i+1, min(i+6, len(indices))):
                        edge_dict['same_level'].append([indices[i], indices[j]])
                        edge_dict['same_level'].append([indices[j], indices[i]])
        
        # 添加同子本体边（采样）
        print("  添加同子本体连接...")
        for namespace, go_ids in self.namespace_terms.items():
            indices = [self.go_id_to_idx[go_id] for go_id in go_ids 
                      if go_id in self.go_id_to_idx]
            if len(indices) > 1:
                np.random.shuffle(indices)
                for i in range(len(indices)):
                    for j in range(i+1, min(i+6, len(indices))):
                        edge_dict['same_namespace'].append([indices[i], indices[j]])
                        edge_dict['same_namespace'].append([indices[j], indices[i]])
        
        return edge_dict
    
    def _generate_hierarchical_features(self):
        """生成层次感知的节点特征"""
        print("  生成节点特征...")
        
        num_nodes = len(self.go_terms)
        namespace_dim = 3
        depth_dim = 1
        structural_dim = 10
        
        features = torch.zeros((num_nodes, namespace_dim + depth_dim + structural_dim))
        
        # 计算深度
        depths = self.analyzer.compute_all_depths()
        
        for go_id, idx in self.go_id_to_idx.items():
            term = self.go_terms[go_id]
            
            # Namespace one-hot
            if 'namespace' in term:
                ns = term['namespace']
                namespaces = ['biological_process', 'molecular_function', 'cellular_component']
                if ns in namespaces:
                    features[idx, namespaces.index(ns)] = 1.0
            
            # 归一化深度
            depth = depths.get(go_id, 0)
            max_depth = 20
            features[idx, namespace_dim] = min(depth / max_depth, 1.0)
            
            # 结构特征
            in_degree = sum(1 for s, t, _ in self.relationships if t == go_id)
            out_degree = sum(1 for s, t, _ in self.relationships if s == go_id)
            
            features[idx, namespace_dim + depth_dim] = min(in_degree / 50, 1.0)
            features[idx, namespace_dim + depth_dim + 1] = min(out_degree / 10, 1.0)
            
            # 名称长度特征
            if 'name' in term:
                name = term['name']
                features[idx, namespace_dim + depth_dim + 2] = min(len(name) / 100, 1.0)
                features[idx, namespace_dim + depth_dim + 3] = len(name.split()) / 10
            
            # 定义长度特征
            if 'def' in term:
                definition = term['def']
                features[idx, namespace_dim + depth_dim + 4] = min(len(definition) / 500, 1.0)
        
        return features
    
    def _get_namespace_tensor(self):
        """获取namespace标签"""
        namespace_map = {
            'biological_process': 0,
            'molecular_function': 1,
            'cellular_component': 2
        }
        
        labels = torch.zeros(len(self.go_terms), dtype=torch.long)
        for go_id, idx in self.go_id_to_idx.items():
            term = self.go_terms[go_id]
            if 'namespace' in term:
                ns = term['namespace']
                if ns in namespace_map:
                    labels[idx] = namespace_map[ns]
        
        return labels
    
    def _get_depth_tensor(self):
        """获取深度张量"""
        depths = self.analyzer.compute_all_depths()
        depth_tensor = torch.zeros(len(self.go_terms), dtype=torch.float)
        
        for go_id, idx in self.go_id_to_idx.items():
            depth_tensor[idx] = depths.get(go_id, 0)
        
        return depth_tensor
    
    def _get_degree_tensor(self):
        """获取节点度数张量"""
        degrees = torch.zeros(len(self.go_terms), dtype=torch.long)
        
        for go_id, idx in self.go_id_to_idx.items():
            in_degree = sum(1 for s, t, _ in self.relationships if t == go_id)
            degrees[idx] = in_degree
        
        return degrees

# ==================== 4. 边Dropout模块 ====================
class EdgeDropout(nn.Module):
    """度感知的边dropout"""
    
    def __init__(self, p=0.2, hub_threshold=50, hub_p=0.5):
        super().__init__()
        self.p = p
        self.hub_threshold = hub_threshold
        self.hub_p = hub_p
    
    def forward(self, edge_index, node_degrees=None):
        if not self.training:
            return edge_index
        
        device = edge_index.device
        
        if node_degrees is not None:
            # Hub节点的边使用更高的dropout
            target_degrees = node_degrees[edge_index[1]]
            is_hub = target_degrees > self.hub_threshold
            
            drop_probs = torch.where(
                is_hub,
                torch.full_like(target_degrees, self.hub_p, dtype=torch.float),
                torch.full_like(target_degrees, self.p, dtype=torch.float)
            )
            
            mask = torch.rand(edge_index.size(1), device=device) > drop_probs
        else:
            mask = torch.rand(edge_index.size(1), device=device) > self.p
        
        return edge_index[:, mask]

# ==================== 5. 修复的层次感知GNN模型 ====================
class OptimizedHierarchicalGOGNN(nn.Module):
    """优化的层次感知GNN模型（修复版）"""
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        input_dim = config['input_dim']
        hidden_dim = config['model']['hidden_dim']
        output_dim = config['model']['output_dim']
        num_layers = config['model']['num_layers']
        dropout = config['model']['dropout']
        use_attention = config['model'].get('use_attention', True)
        attention_heads = config['model'].get('attention_heads', 4)
        use_residual = config['model'].get('residual', True)
        
        self.num_layers = num_layers
        self.use_residual = use_residual
        
        # 边Dropout
        edge_dropout_p = config['model'].get('edge_dropout', 0.2)
        if edge_dropout_p > 0:
            degree_config = config.get('degree_aware', {})
            self.edge_dropout = EdgeDropout(
                p=edge_dropout_p,
                hub_threshold=degree_config.get('hub_threshold', 50),
                hub_p=degree_config.get('hub_dropout', 0.5)
            )
        else:
            self.edge_dropout = None
        
        # 卷积层和归一化层
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # 记录每层的实际输出维度
        self.layer_dims = []
        
        # 第一层（总是使用GCN）
        current_dim = input_dim
        next_dim = hidden_dim
        self.convs.append(self._create_hetero_gcn(current_dim, next_dim))
        self.layer_dims.append(next_dim)
        self.norms.append(self._create_norm(next_dim, config['model'].get('norm', 'layer')))
        current_dim = next_dim
        
        # 中间层
        for i in range(1, num_layers - 1):
            if use_attention:
                # GAT层：输出维度 = out_channels * heads（如果concat=True）
                out_channels_per_head = hidden_dim // attention_heads
                conv = self._create_hetero_gat(current_dim, out_channels_per_head, 
                                               attention_heads, concat=True)
                # GAT拼接后的实际输出维度
                actual_out_dim = out_channels_per_head * attention_heads
            else:
                conv = self._create_hetero_gcn(current_dim, hidden_dim)
                actual_out_dim = hidden_dim
            
            self.convs.append(conv)
            self.layer_dims.append(actual_out_dim)
            self.norms.append(self._create_norm(actual_out_dim, config['model'].get('norm', 'layer')))
            current_dim = actual_out_dim
        
        # 最后一层
        if num_layers > 1:
            if use_attention:
                # 最后一层不拼接，heads=1
                conv = self._create_hetero_gat(current_dim, output_dim, 1, concat=False)
            else:
                conv = self._create_hetero_gcn(current_dim, output_dim)
            self.convs.append(conv)
            self.layer_dims.append(output_dim)
        
        # Residual投影层
        if use_residual and num_layers > 1:
            self.residual_projs = nn.ModuleList()
            prev_dim = input_dim
            for i in range(num_layers - 1):
                curr_layer_dim = self.layer_dims[i]
                if prev_dim != curr_layer_dim:
                    self.residual_projs.append(nn.Linear(prev_dim, curr_layer_dim))
                else:
                    self.residual_projs.append(nn.Identity())
                prev_dim = curr_layer_dim
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Activation
        activation = config['model'].get('activation', 'gelu')
        if activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu
        
        # 子本体投影
        self.namespace_projections = nn.ModuleDict({
            'bp': nn.Linear(output_dim, output_dim),
            'mf': nn.Linear(output_dim, output_dim),
            'cc': nn.Linear(output_dim, output_dim)
        })
    
    def _create_norm(self, dim, norm_type):
        """创建归一化层"""
        if norm_type == 'batch':
            return nn.BatchNorm1d(dim)
        elif norm_type == 'layer':
            return nn.LayerNorm(dim)
        else:
            return nn.Identity()
    
    def _create_hetero_gcn(self, in_channels, out_channels):
        """创建异构GCN"""
        conv_dict = {}
        edge_types = ['is_a', 'part_of', 'regulates', 'same_level', 'same_namespace']
        for edge_type in edge_types:
            conv_dict[('go_term', edge_type, 'go_term')] = GCNConv(in_channels, out_channels)
        return HeteroConv(conv_dict, aggr='mean')
    
    def _create_hetero_gat(self, in_channels, out_channels, heads, concat):
        """创建异构GAT"""
        conv_dict = {}
        edge_types = ['is_a', 'part_of', 'regulates', 'same_level', 'same_namespace']
        attention_dropout = self.config['model'].get('attention_dropout', 0.3)
        
        for edge_type in edge_types:
            conv_dict[('go_term', edge_type, 'go_term')] = GATConv(
                in_channels, out_channels, heads=heads, concat=concat, 
                dropout=attention_dropout
            )
        return HeteroConv(conv_dict, aggr='mean')
    
    def forward(self, x_dict, edge_index_dict, namespace_labels, depth, node_degrees=None):
        x = x_dict['go_term']
        
        # 应用边dropout
        if self.edge_dropout is not None and self.training:
            edge_index_dict_dropped = {}
            for edge_type, edge_index in edge_index_dict.items():
                edge_index_dict_dropped[edge_type] = self.edge_dropout(
                    edge_index, node_degrees=node_degrees
                )
        else:
            edge_index_dict_dropped = edge_index_dict
        
        # GNN层
        for i in range(self.num_layers):
            x_prev = x
            
            # 卷积
            x_dict_temp = {'go_term': x}
            x_dict_temp = self.convs[i](x_dict_temp, edge_index_dict_dropped)
            x = x_dict_temp['go_term']
            
            # 残差连接（除了最后一层）
            if self.use_residual and i < self.num_layers - 1:
                x_prev_projected = self.residual_projs[i](x_prev)
                x = x + x_prev_projected
            
            # 归一化 + 激活 + Dropout（除了最后一层）
            if i < self.num_layers - 1:
                x = self.norms[i](x)
                x = self.activation(x)
                x = self.dropout(x)
        
        # 子本体投影
        x = self._apply_namespace_projection(x, namespace_labels)
        
        return x
    
    def _apply_namespace_projection(self, x, namespace_labels):
        """应用子本体特定的投影"""
        x_projected = torch.zeros_like(x)
        namespace_map = {0: 'bp', 1: 'mf', 2: 'cc'}
        
        for ns_id, ns_key in namespace_map.items():
            mask = (namespace_labels == ns_id)
            if mask.any():
                x_projected[mask] = self.namespace_projections[ns_key](x[mask])
        
        return x_projected

# ==================== 6. 修复的训练器 ====================
class OptimizedTrainer:
    """优化的训练器（修复维度不匹配问题）"""
    
    def __init__(self, model, data, config, device='cpu'):
        self.model = model.to(device)
        self.data = data.to(device)
        self.config = config
        self.device = device
        self.current_epoch = 0
        
        # Optimizer
        optimizer_type = config['training']['optimizer']
        if optimizer_type == 'adamw':
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config['training']['learning_rate'],
                weight_decay=config['training']['weight_decay']
            )
        else:
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config['training']['learning_rate'],
                weight_decay=config['training']['weight_decay']
            )
        
        # Scheduler
        self._setup_scheduler()
        
        # 梯度裁剪
        self.gradient_clip = config['training'].get('gradient_clip', None)
        
        # 训练历史
        self.train_history = {
            'loss': [],
            'learning_rate': [],
            'loss_components': []
        }
    
    def _setup_scheduler(self):
        """设置学习率调度器"""
        scheduler_type = self.config['training'].get('scheduler', 'cosine')
        
        if scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['num_epochs'],
                eta_min=self.config['training'].get('min_lr', 1e-6)
            )
        elif scheduler_type == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=self.config['training'].get('patience', 20),
                verbose=True
            )
        else:
            self.scheduler = None
        
        # Warmup
        self.warmup_epochs = self.config['training'].get('warmup_epochs', 0)
        if self.warmup_epochs > 0:
            self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.warmup_epochs
            )
    
    def train(self, num_epochs=None):
        """训练模型"""
        if num_epochs is None:
            num_epochs = self.config['training']['num_epochs']
        
        print(f"\n开始训练 ({num_epochs} epochs)...")
        
        best_loss = float('inf')
        patience_counter = 0
        max_patience = self.config['training'].get('patience', 50)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            self.model.train()
            self.optimizer.zero_grad()
            
            # 准备输入
            x_dict = {'go_term': self.data['go_term'].x}
            edge_index_dict = {
                key: self.data[key].edge_index 
                for key in self.data.edge_types
            }
            
            # 前向传播
            embeddings = self.model(
                x_dict,
                edge_index_dict,
                self.data['go_term'].namespace,
                self.data['go_term'].depth,
                self.data['go_term'].node_degrees
            )
            
            # 计算损失
            loss, loss_components = self._compute_hierarchical_loss(
                embeddings,
                edge_index_dict,
                self.data['go_term'].namespace,
                self.data['go_term'].depth,
                self.data['go_term'].node_degrees
            )
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            self.optimizer.step()
            
            # 学习率调度
            current_lr = self.optimizer.param_groups[0]['lr']
            if epoch < self.warmup_epochs:
                self.warmup_scheduler.step()
            elif self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(loss)
                else:
                    self.scheduler.step()
            
            # 记录历史
            self.train_history['loss'].append(loss.item())
            self.train_history['learning_rate'].append(current_lr)
            self.train_history['loss_components'].append(loss_components)
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1:3d}/{num_epochs}, '
                      f'Loss: {loss.item():.4f}, '
                      f'LR: {current_lr:.6f}')
            
            # 打印损失组件（每50个epoch）
            if (epoch + 1) % 50 == 0:
                print(f'  损失组件: {loss_components}')
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= max_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        print(f"\n✓ 训练完成! 最佳损失: {best_loss:.4f}")
        
        return self.get_embeddings()
    
    def _compute_hierarchical_loss(self, embeddings, edge_index_dict, 
                                   namespace_labels, depth, node_degrees):
        """修复的层次感知损失函数"""
        loss_weights = self.config.get('loss_weights', {})
        edge_weights = self.config.get('edge_weights', {})
        
        total_loss = 0.0
        loss_components = {}
        
        # 1. 链接预测损失（主要损失）
        link_loss_total = 0.0
        for edge_type_tuple, edge_index in edge_index_dict.items():
            if edge_index.size(1) > 0:
                edge_type_str = edge_type_tuple[1]
                loss_link = self._link_prediction_loss(embeddings, edge_index)
                edge_weight = edge_weights.get(edge_type_str, 1.0)
                link_loss_total += edge_weight * loss_link
        
        total_loss += link_loss_total
        loss_components['link'] = f"{link_loss_total.item():.4f}"
        
        # 2. 对比损失（防止collapse）
        contrastive_loss = self._contrastive_loss(embeddings, namespace_labels)
        total_loss += 0.5 * contrastive_loss
        loss_components['contrast'] = f"{contrastive_loss.item():.4f}"
        
        # 3. 层次一致性损失
        if ('go_term', 'is_a', 'go_term') in edge_index_dict:
            loss_hierarchy = self._hierarchy_consistency_loss(
                embeddings,
                edge_index_dict[('go_term', 'is_a', 'go_term')]
            )
            total_loss += 0.1 * loss_hierarchy
            loss_components['hier'] = f"{loss_hierarchy.item():.4f}"
        
        # 4. 方差正则化（防止collapse）
        variance_loss = self._variance_regularization(embeddings)
        total_loss += 0.1 * variance_loss
        loss_components['var'] = f"{variance_loss.item():.4f}"
        
        # 5. 子本体分离损失
        separation_loss = self._namespace_separation_loss(embeddings, namespace_labels)
        total_loss += 0.3 * separation_loss
        loss_components['sep'] = f"{separation_loss.item():.4f}"
        
        return total_loss, loss_components
    
    def _contrastive_loss(self, embeddings, namespace_labels):
        """对比损失：同子本体内的应该相似，不同子本体的应该不同"""
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        
        # 随机采样
        n_samples = min(1000, embeddings.size(0))
        indices = torch.randperm(embeddings.size(0), device=embeddings.device)[:n_samples]
        
        sampled_embs = embeddings_norm[indices]
        sampled_labels = namespace_labels[indices]
        
        # 计算相似度矩阵
        similarity = torch.mm(sampled_embs, sampled_embs.t())
        
        # 创建标签矩阵
        labels_expanded = sampled_labels.unsqueeze(1)
        positive_mask = (labels_expanded == labels_expanded.t()).float()
        
        # 去除对角线
        diagonal_mask = torch.eye(n_samples, device=embeddings.device)
        positive_mask = positive_mask * (1 - diagonal_mask)
        negative_mask = (1 - positive_mask) * (1 - diagonal_mask)
        
        # 正样本损失
        if positive_mask.sum() > 0:
            positive_loss = -torch.log(torch.sigmoid(similarity * 10) + 1e-8)
            positive_loss = (positive_loss * positive_mask).sum() / (positive_mask.sum() + 1e-8)
        else:
            positive_loss = torch.tensor(0.0, device=embeddings.device)
        
        # 负样本损失
        if negative_mask.sum() > 0:
            negative_loss = -torch.log(1 - torch.sigmoid(similarity * 10) + 1e-8)
            negative_loss = (negative_loss * negative_mask).sum() / (negative_mask.sum() + 1e-8)
        else:
            negative_loss = torch.tensor(0.0, device=embeddings.device)
        
        return positive_loss + negative_loss
    
    def _variance_regularization(self, embeddings):
        """方差正则化：鼓励embeddings在每个维度上有足够的方差"""
        variance_per_dim = torch.var(embeddings, dim=0)
        target_variance = 1.0
        variance_loss = torch.mean(F.relu(target_variance - variance_per_dim))
        return variance_loss
    
    def _namespace_separation_loss(self, embeddings, namespace_labels):
        """改进的子本体分离损失"""
        loss = 0.0
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        
        # 计算每个子本体的中心
        centers = []
        for ns_id in range(3):
            mask = (namespace_labels == ns_id)
            if mask.sum() > 0:
                center = embeddings_norm[mask].mean(dim=0)
                centers.append(center)
        
        if len(centers) < 2:
            return torch.tensor(0.0, device=embeddings.device)
        
        centers = torch.stack(centers)
        
        # 最大化不同中心之间的距离
        center_similarity = torch.mm(centers, centers.t())
        n_centers = centers.size(0)
        mask = ~torch.eye(n_centers, dtype=torch.bool, device=embeddings.device)
        inter_center_sim = center_similarity[mask]
        
        separation_loss = inter_center_sim.mean()
        
        # 最小化每个子本体内部的方差
        intra_variance = 0.0
        for ns_id, center in enumerate(centers):
            mask = (namespace_labels == ns_id)
            if mask.sum() > 1:
                ns_embs = embeddings_norm[mask]
                distances = 1 - torch.mm(ns_embs, center.unsqueeze(1)).squeeze()
                intra_variance += distances.mean()
        
        intra_variance /= len(centers)
        
        return separation_loss - 0.3 * intra_variance
    
    def _link_prediction_loss(self, embeddings, edge_index):
        """修复的链接预测损失 - 关键修复！"""
        if edge_index.size(1) == 0:
            return torch.tensor(0.0, device=self.device)
        
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        
        # 采样正样本以匹配负样本数量
        num_pos = edge_index.size(1)
        num_samples = min(num_pos, 2000)  # 最多采样2000条边
        
        if num_pos > num_samples:
            # 随机采样正样本
            sample_indices = torch.randperm(num_pos, device=self.device)[:num_samples]
            sampled_pos_edges = edge_index[:, sample_indices]
        else:
            sampled_pos_edges = edge_index
            num_samples = num_pos
        
        # 正样本得分
        pos_score = (embeddings_norm[sampled_pos_edges[0]] * 
                    embeddings_norm[sampled_pos_edges[1]]).sum(dim=1)
        
        # 负采样 - 数量与正样本相同
        neg_edge_index = self._negative_sampling(sampled_pos_edges, embeddings.size(0), num_samples)
        neg_score = (embeddings_norm[neg_edge_index[0]] * 
                    embeddings_norm[neg_edge_index[1]]).sum(dim=1)
        
        # 现在pos_score和neg_score的维度相同，可以安全计算loss
        margin = 0.5
        loss = torch.mean(F.relu(margin - pos_score + neg_score))
        
        return loss
    
    def _hierarchy_consistency_loss(self, embeddings, is_a_edges):
        """修改：降低约束强度"""
        if is_a_edges.size(1) == 0:
            return torch.tensor(0.0, device=self.device)
        
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        
        # 采样以提高效率
        num_edges = is_a_edges.size(1)
        if num_edges > 5000:
            sample_indices = torch.randperm(num_edges, device=self.device)[:5000]
            is_a_edges = is_a_edges[:, sample_indices]
        
        child_emb = embeddings_norm[is_a_edges[0]]
        parent_emb = embeddings_norm[is_a_edges[1]]
        
        similarity = (child_emb * parent_emb).sum(dim=1)
        target_similarity = 0.7
        
        loss = F.relu(target_similarity - similarity).mean()
        
        return loss
    
    def _negative_sampling(self, pos_edge_index, num_nodes, num_neg):
        """负采样"""
        neg_edges = []
        pos_edges_set = set(map(tuple, pos_edge_index.t().cpu().numpy()))
        
        max_attempts = num_neg * 10
        attempts = 0
        
        while len(neg_edges) < num_neg and attempts < max_attempts:
            src = torch.randint(0, num_nodes, (1,), device=self.device).item()
            tgt = torch.randint(0, num_nodes, (1,), device=self.device).item()
            if (src, tgt) not in pos_edges_set and src != tgt:
                neg_edges.append([src, tgt])
            attempts += 1
        
        # 如果采样不足，用随机边填充
        while len(neg_edges) < num_neg:
            src = torch.randint(0, num_nodes, (1,), device=self.device).item()
            tgt = torch.randint(0, num_nodes, (1,), device=self.device).item()
            if src != tgt:
                neg_edges.append([src, tgt])
        
        return torch.tensor(neg_edges, dtype=torch.long, device=self.device).t()
    
    def get_embeddings(self):
        """获取embeddings"""
        self.model.eval()
        with torch.no_grad():
            x_dict = {'go_term': self.data['go_term'].x}
            edge_index_dict = {
                key: self.data[key].edge_index 
                for key in self.data.edge_types
            }
            embeddings = self.model(
                x_dict,
                edge_index_dict,
                self.data['go_term'].namespace,
                self.data['go_term'].depth,
                self.data['go_term'].node_degrees
            )
        return embeddings.cpu().numpy()
    
    def plot_training_history(self, save_path='training_history.png'):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.train_history['loss'])
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(alpha=0.3)
        
        ax2.plot(self.train_history['learning_rate'])
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(alpha=0.3)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 训练历史图已保存: {save_path}")

# ==================== 7. 配置管理 ====================
class ConfigManager:
    """配置管理器"""
    
    BALANCED_CONFIG = {
        'model': {
            'num_layers': 5,
            'hidden_dim': 512,
            'output_dim': 256,
            'dropout': 0.4,
            'edge_dropout': 0.2,
            'attention_dropout': 0.3,
            'use_attention': True,
            'attention_heads': 4,
            'residual': True,
            'activation': 'gelu',
            'norm': 'layer',
        },
        'training': {
            'num_epochs': 500,
            'learning_rate': 0.005,
            'weight_decay': 5e-4,
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'warmup_epochs': 30,
            'min_lr': 1e-6,
            'gradient_clip': 1.0,
            'patience': 50,
        },
        'loss_weights': {
            'link_prediction': 1.0,
            'namespace_similarity': 0.5,
            'hierarchy_consistency': 0.4,
            'namespace_balance': {
                'bp': 1.0,
                'mf': 1.5,
                'cc': 2.0,
            }
        },
        'edge_weights': {
            'is_a': 2.5,
            'part_of': 1.5,
            'regulates': 1.0,
            'same_level': 0.3,
            'same_namespace': 0.3,
        },
        'degree_aware': {
            'enabled': True,
            'hub_threshold': 50,
            'hub_dropout': 0.5,
            'leaf_boost': 1.2,
        }
    }
    
    CONSERVATIVE_CONFIG = {
        'model': {
            'num_layers': 4,
            'hidden_dim': 256,
            'output_dim': 128,
            'dropout': 0.3,
            'edge_dropout': 0.1,
            'use_attention': True,
            'attention_heads': 4,
            'residual': True,
            'activation': 'relu',
            'norm': 'batch',
        },
        'training': {
            'num_epochs': 200,
            'learning_rate': 0.01,
            'weight_decay': 5e-4,
            'optimizer': 'adam',
            'scheduler': 'plateau',
            'warmup_epochs': 20,
            'gradient_clip': 1.0,
            'patience': 30,
        },
        'loss_weights': {
            'link_prediction': 1.0,
            'namespace_similarity': 0.3,
            'hierarchy_consistency': 0.2,
        },
        'edge_weights': {
            'is_a': 2.0,
            'part_of': 1.0,
            'regulates': 1.0,
            'same_level': 0.5,
            'same_namespace': 0.5,
        },
        'degree_aware': {
            'enabled': False,
        }
    }
    
    AGGRESSIVE_CONFIG = {
        'model': {
            'num_layers': 7,
            'hidden_dim': 768,
            'output_dim': 256,
            'dropout': 0.5,
            'edge_dropout': 0.3,
            'attention_dropout': 0.4,
            'use_attention': True,
            'attention_heads': 8,
            'residual': True,
            'activation': 'gelu',
            'norm': 'layer',
        },
        'training': {
            'num_epochs': 500,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'warmup_epochs': 50,
            'min_lr': 1e-7,
            'gradient_clip': 1.0,
            'patience': 60,
        },
        'loss_weights': {
            'link_prediction': 1.0,
            'namespace_similarity': 0.6,
            'hierarchy_consistency': 0.5,
            'namespace_balance': {
                'bp': 1.0,
                'mf': 2.0,
                'cc': 3.0,
            }
        },
        'edge_weights': {
            'is_a': 3.0,
            'part_of': 2.0,
            'regulates': 1.5,
            'same_level': 0.2,
            'same_namespace': 0.2,
        },
        'degree_aware': {
            'enabled': True,
            'hub_threshold': 40,
            'hub_dropout': 0.6,
            'leaf_boost': 1.5,
        }
    }
    
    @staticmethod
    def get_config(mode='balanced'):
        """获取配置"""
        configs = {
            'conservative': ConfigManager.CONSERVATIVE_CONFIG,
            'balanced': ConfigManager.BALANCED_CONFIG,
            'aggressive': ConfigManager.AGGRESSIVE_CONFIG,
        }
        return configs.get(mode, ConfigManager.BALANCED_CONFIG).copy()

# ==================== 8. 结果分析器 ====================
class EmbeddingAnalyzer:
    """Embedding质量分析器"""
    
    def __init__(self, embeddings, data, go_id_to_idx, namespace_terms, output_dir):
        self.embeddings = embeddings
        self.data = data
        self.go_id_to_idx = go_id_to_idx
        self.namespace_terms = namespace_terms
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
    
    def analyze(self):
        """运行完整分析"""
        print("\n" + "="*60)
        print("Embedding质量分析")
        print("="*60)
        
        self._analyze_namespace_separation()
        self._visualize_embeddings()
        self._show_similarity_examples()
        
        print("="*60)
    
    def _analyze_namespace_separation(self):
        """分析子本体分离度"""
        print("\n[1] 子本体分离度分析")
        
        try:
            from sklearn.metrics import silhouette_score
            
            namespace_labels = self.data['go_term'].namespace.cpu().numpy()
            
            if len(set(namespace_labels)) > 1:
                score = silhouette_score(self.embeddings, namespace_labels)
                print(f"  Silhouette Score: {score:.4f}")
                print(f"  {'✓ 良好分离' if score > 0.3 else '⚠ 需要改进'}")
        except ImportError:
            print("  需要安装sklearn: pip install scikit-learn")
    
    def _visualize_embeddings(self):
        """可视化embeddings"""
        print("\n[2] 可视化embeddings (使用t-SNE)")
        
        try:
            from sklearn.manifold import TSNE
            
            sample_size = min(5000, len(self.embeddings))
            indices = np.random.choice(len(self.embeddings), sample_size, replace=False)
            
            embeddings_sample = self.embeddings[indices]
            namespace_labels = self.data['go_term'].namespace.cpu().numpy()[indices]
            
            print("  运行t-SNE (可能需要几分钟)...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            embeddings_2d = tsne.fit_transform(embeddings_sample)
            
            plt.figure(figsize=(10, 8))
            
            namespace_names = ['BP', 'MF', 'CC']
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            
            for ns_id, (name, color) in enumerate(zip(namespace_names, colors)):
                mask = namespace_labels == ns_id
                if mask.any():
                    plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                              c=color, label=name, alpha=0.6, s=10)
            
            plt.legend()
            plt.title('GO Term Embeddings (t-SNE)')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.tight_layout()
            
            save_path = os.path.join(self.output_dir, 'embeddings_tsne.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ 可视化已保存: {save_path}")
            
        except ImportError:
            print("  需要安装sklearn: pip install scikit-learn")
    
    def _show_similarity_examples(self):
        """显示相似度示例"""
        print("\n[3] 相似度示例")
        
        sample_go_ids = list(self.go_id_to_idx.keys())[:100]
        query_go_id = np.random.choice(sample_go_ids)
        
        query_idx = self.go_id_to_idx[query_go_id]
        query_emb = self.embeddings[query_idx]
        
        similarities = np.dot(self.embeddings, query_emb) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-8
        )
        
        top_indices = np.argsort(similarities)[::-1][1:6]
        
        idx_to_go_id = {v: k for k, v in self.go_id_to_idx.items()}
        
        print(f"\n  查询GO term: {query_go_id}")
        print(f"  最相似的5个GO terms:")
        for rank, idx in enumerate(top_indices, 1):
            go_id = idx_to_go_id[idx]
            sim = similarities[idx]
            print(f"    {rank}. {go_id} (相似度: {sim:.4f})")

# ==================== 9. Embedding诊断工具 ====================
def diagnose_embeddings(embeddings, metadata, output_dir):
    """诊断embedding质量"""
    print("\n" + "="*80)
    print("Embedding质量诊断")
    print("="*80)
    
    # 1. 检查embedding的基本统计
    print("\n[1] 基本统计")
    print(f"  Embedding形状: {embeddings.shape}")
    print(f"  均值: {np.mean(embeddings):.6f}")
    print(f"  标准差: {np.std(embeddings):.6f}")
    print(f"  最小值: {np.min(embeddings):.6f}")
    print(f"  最大值: {np.max(embeddings):.6f}")
    
    # 2. 检查是否所有embedding都相同
    print("\n[2] 多样性检查")
    unique_rows = np.unique(embeddings, axis=0)
    print(f"  唯一embedding数量: {len(unique_rows)} / {len(embeddings)}")
    
    if len(unique_rows) == 1:
        print("  ❌ 严重问题: 所有embedding完全相同!")
        return False
    
    # 3. 检查embedding的范数分布
    print("\n[3] 范数分布")
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"  平均范数: {np.mean(norms):.6f}")
    print(f"  范数标准差: {np.std(norms):.6f}")
    print(f"  范数最小值: {np.min(norms):.6f}")
    print(f"  范数最大值: {np.max(norms):.6f}")
    
    if np.std(norms) < 0.01:
        print("  ⚠️ 警告: 所有embedding的范数几乎相同")
    
    # 4. 检查相似度分布
    print("\n[4] 相似度分布")
    n_samples = min(1000, len(embeddings))
    indices = np.random.choice(len(embeddings), n_samples, replace=False)
    sample_embs = embeddings[indices]
    
    # 归一化
    sample_embs_norm = sample_embs / (np.linalg.norm(sample_embs, axis=1, keepdims=True) + 1e-8)
    
    # 计算相似度矩阵
    similarity_matrix = np.dot(sample_embs_norm, sample_embs_norm.T)
    
    # 去除对角线
    mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
    similarities = similarity_matrix[mask]
    
    print(f"  平均相似度: {np.mean(similarities):.6f}")
    print(f"  相似度标准差: {np.std(similarities):.6f}")
    print(f"  最小相似度: {np.min(similarities):.6f}")
    print(f"  最大相似度: {np.max(similarities):.6f}")
    print(f"  相似度中位数: {np.median(similarities):.6f}")
    
    # 统计高相似度的比例
    high_sim_ratio = np.sum(similarities > 0.99) / len(similarities)
    print(f"  相似度>0.99的比例: {high_sim_ratio*100:.2f}%")
    
    if high_sim_ratio > 0.5:
        print("  ❌ 严重问题: 超过50%的embedding对相似度>0.99!")
        return False
    elif high_sim_ratio > 0.1:
        print("  ⚠️ 警告: 超过10%的embedding对相似度>0.99")
    else:
        print("  ✓ 相似度分布合理")
    
    # 5. 绘制相似度分布直方图
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(similarities, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Distribution of Pairwise Similarities')
    plt.axvline(x=np.mean(similarities), color='r', linestyle='--', label=f'Mean: {np.mean(similarities):.3f}')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(norms, bins=50, edgecolor='black', alpha=0.7, color='orange')
    plt.xlabel('L2 Norm')
    plt.ylabel('Frequency')
    plt.title('Distribution of Embedding Norms')
    plt.axvline(x=np.mean(norms), color='r', linestyle='--', label=f'Mean: {np.mean(norms):.3f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'embedding_diagnostics.png'), dpi=300)
    plt.close()
    
    print(f"\n  ✓ 诊断图表已保存: {os.path.join(output_dir, 'embedding_diagnostics.png')}")
    
    # 6. 检查子本体分离
    print("\n[5] 子本体分离度")
    try:
        from sklearn.metrics import silhouette_score
        
        go_id_to_idx = metadata['go_id_to_idx']
        namespace_terms = metadata['namespace_terms']
        
        namespace_labels = np.zeros(len(embeddings), dtype=int)
        namespace_map = {'biological_process': 0, 'molecular_function': 1, 'cellular_component': 2}
        
        for ns, ns_id in namespace_map.items():
            for go_id in namespace_terms[ns]:
                if go_id in go_id_to_idx:
                    idx = go_id_to_idx[go_id]
                    namespace_labels[idx] = ns_id
        
        if len(set(namespace_labels)) > 1:
            score = silhouette_score(embeddings, namespace_labels)
            print(f"  Silhouette Score: {score:.4f}")
            
            if score < 0.1:
                print("  ❌ 严重问题: 子本体完全没有分离!")
                return False
            elif score < 0.3:
                print("  ⚠️ 警告: 子本体分离度较差")
            else:
                print("  ✓ 子本体分离度良好")
    except:
        pass
    
    print("\n" + "="*80)
    return True

# ==================== 10. 主程序 ====================
def main(obo_file_path, output_dir='./go_gnn_results', config_mode='balanced'):
    """主函数"""
    print("="*80)
    print("GO Term GNN - 完整训练流程（修复版v3）")
    print("="*80)
    print(f"\n配置:")
    print(f"  OBO文件: {obo_file_path}")
    print(f"  输出目录: {output_dir}")
    print(f"  配置模式: {config_mode}")
    print(f"  设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. 解析
        print("\n" + "="*80)
        print("[步骤 1/7] 解析OBO文件")
        print("="*80)
        parser = OBOParser(obo_file_path)
        go_terms, relationships, namespace_terms = parser.parse()
        
        # 2. 构建图
        print("\n" + "="*80)
        print("[步骤 2/7] 构建层次感知图")
        print("="*80)
        builder = HierarchicalGOGraphBuilder(go_terms, relationships, namespace_terms)
        data, go_id_to_idx, idx_to_go_id = builder.build_hierarchical_graph()
        
        # 3. 创建模型
        print("\n" + "="*80)
        print("[步骤 3/7] 创建GNN模型")
        print("="*80)
        
        config = ConfigManager.get_config(config_mode)
        config['input_dim'] = data['go_term'].x.size(1)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = OptimizedHierarchicalGOGNN(config)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ 模型创建完成")
        print(f"  总参数: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  配置: {config_mode}")
        
        # 4. 训练
        print("\n" + "="*80)
        print("[步骤 4/7] 训练模型")
        print("="*80)
        
        trainer = OptimizedTrainer(model, data, config, device=device)
        embeddings = trainer.train()
        
        # 5. 保存结果
        print("\n" + "="*80)
        print("[步骤 5/7] 保存结果")
        print("="*80)
        
        embeddings_path = os.path.join(output_dir, 'go_term_embeddings.npy')
        np.save(embeddings_path, embeddings)
        print(f"✓ Embeddings已保存: {embeddings_path}")
        
        metadata = {
            'go_id_to_idx': go_id_to_idx,
            'idx_to_go_id': idx_to_go_id,
            'namespace_terms': {k: list(v) for k, v in namespace_terms.items()},
            'config': config,
            'num_nodes': len(go_terms),
            'num_edges': len(relationships),
            'embedding_dim': embeddings.shape[1],
        }
        
        metadata_path = os.path.join(output_dir, 'metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"✓ 元数据已保存: {metadata_path}")
        
        trainer.plot_training_history(os.path.join(output_dir, 'training_history.png'))
        
        # 6. 分析结果
        print("\n" + "="*80)
        print("[步骤 6/7] 分析结果")
        print("="*80)
        
        analyzer = EmbeddingAnalyzer(embeddings, data, go_id_to_idx, 
                                    namespace_terms, output_dir)
        analyzer.analyze()
        
        # 7. 诊断
        print("\n" + "="*80)
        print("[步骤 7/7] 诊断Embedding质量")
        print("="*80)
        
        is_healthy = diagnose_embeddings(embeddings, metadata, output_dir)
        
        if not is_healthy:
            print("\n⚠️ 警告: Embedding质量存在问题，建议:")
            print("  1. 检查训练曲线是否正常下降")
            print("  2. 尝试降低学习率")
            print("  3. 增加对比损失的权重")
            print("  4. 减少层数或降低模型复杂度")
        
        # 最终总结
        print("\n" + "="*80)
        print("训练完成!")
        print("="*80)
        print(f"\n输出文件:")
        print(f"  ├─ go_term_embeddings.npy  ({embeddings.shape[0]} x {embeddings.shape[1]})")
        print(f"  ├─ metadata.pkl")
        print(f"  ├─ training_history.png")
        print(f"  ├─ embeddings_tsne.png")
        print(f"  └─ embedding_diagnostics.png")
        print(f"\n所有文件保存在: {output_dir}")
        print("="*80 + "\n")
        
        return embeddings, metadata
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

# ==================== 11. 工具函数 ====================
def load_embeddings(output_dir='./go_gnn_results'):
    """加载训练好的embeddings"""
    embeddings = np.load(os.path.join(output_dir, 'go_term_embeddings.npy'))
    
    with open(os.path.join(output_dir, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    
    return embeddings, metadata

def get_embedding(go_id, embeddings, metadata):
    """获取特定GO term的embedding"""
    go_id_to_idx = metadata['go_id_to_idx']
    if go_id in go_id_to_idx:
        idx = go_id_to_idx[go_id]
        return embeddings[idx]
    else:
        return None

def find_similar_terms(go_id, embeddings, metadata, k=10):
    """找到最相似的k个GO terms"""
    go_id_to_idx = metadata['go_id_to_idx']
    idx_to_go_id = metadata['idx_to_go_id']
    
    if go_id not in go_id_to_idx:
        return []
    
    query_idx = go_id_to_idx[go_id]
    query_emb = embeddings[query_idx]
    
    similarities = np.dot(embeddings, query_emb) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-8
    )
    
    top_indices = np.argsort(similarities)[::-1][1:k+1]
    
    results = []
    for idx in top_indices:
        results.append({
            'go_id': idx_to_go_id[idx],
            'similarity': similarities[idx]
        })
    
    return results

# ==================== 12. 命令行接口 ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='GO Term GNN训练（修复版v3）')
    parser.add_argument('--obo', type=str, required=True,
                       help='OBO文件路径')
    parser.add_argument('--output', type=str, default='./go_gnn_results',
                       help='输出目录')
    parser.add_argument('--config', type=str, default='balanced',
                       choices=['conservative', 'balanced', 'aggressive'],
                       help='配置模式')
    
    args = parser.parse_args()
    
    embeddings, metadata = main(
        obo_file_path=args.obo,
        output_dir=args.output,
        config_mode=args.config
    )
    
    if embeddings is not None:
        print("\n使用示例:")
        print("-" * 60)
        
        sample_go_id = list(metadata['go_id_to_idx'].keys())[0]
        
        emb = get_embedding(sample_go_id, embeddings, metadata)
        print(f"\n1. 获取embedding:")
        print(f"   GO ID: {sample_go_id}")
        print(f"   Embedding shape: {emb.shape}")
        print(f"   前5维: {emb[:5]}")
        
        similar = find_similar_terms(sample_go_id, embeddings, metadata, k=5)
        print(f"\n2. 最相似的5个GO terms:")
        for i, item in enumerate(similar, 1):
            print(f"   {i}. {item['go_id']} (相似度: {item['similarity']:.4f})")
        
        print("\n" + "-" * 60)