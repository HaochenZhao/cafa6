"""
GO Term GNN - 完整实现（修复版）
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

# ==================== 6. 优化的训练器 ====================
class OptimizedTrainer:
    """优化的训练器"""
    
    def __init__(self, model, data, config, device='cpu'):
        self.model = model.to(device)
        self.data = data.to(device)
        self.config = config
        self.device = device
        
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
            'learning_rate': []
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
            loss = self._compute_hierarchical_loss(
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
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1:3d}/{num_epochs}, '
                      f'Loss: {loss.item():.4f}, '
                      f'LR: {current_lr:.6f}')
            
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
        """计算层次感知的损失"""
        loss_weights = self.config.get('loss_weights', {})
        edge_weights = self.config.get('edge_weights', {})
        
        total_loss = 0.0
        
        # 1. 链接预测损失
        for edge_type_tuple, edge_index in edge_index_dict.items():
            if edge_index.size(1) > 0:
                edge_type_str = edge_type_tuple[1]
                loss_link = self._link_prediction_loss(embeddings, edge_index)
                edge_weight = edge_weights.get(edge_type_str, 1.0)
                total_loss += edge_weight * loss_link
        
        # 2. 子本体相似性损失
        ns_balance = loss_weights.get('namespace_balance', {})
        for ns_id in range(3):
            mask = (namespace_labels == ns_id)
            if mask.sum() > 1:
                ns_key = ['bp', 'mf', 'cc'][ns_id]
                ns_weight = ns_balance.get(ns_key, 1.0)
                
                ns_embeddings = embeddings[mask]
                loss_ns = self._namespace_similarity_loss(ns_embeddings)
                total_loss += loss_weights.get('namespace_similarity', 0.5) * ns_weight * loss_ns
        
        # 3. 层次一致性损失
        if ('go_term', 'is_a', 'go_term') in edge_index_dict:
            loss_hierarchy = self._hierarchy_consistency_loss(
                embeddings,
                edge_index_dict[('go_term', 'is_a', 'go_term')]
            )
            total_loss += loss_weights.get('hierarchy_consistency', 0.4) * loss_hierarchy
        
        # 4. 叶子节点正则化
        degree_config = self.config.get('degree_aware', {})
        if degree_config.get('enabled', False):
            leaf_mask = (node_degrees == 0)
            if leaf_mask.sum() > 1:
                leaf_boost = degree_config.get('leaf_boost', 1.0)
                leaf_embs = embeddings[leaf_mask]
                leaf_embs_norm = F.normalize(leaf_embs, p=2, dim=1)
                
                # 采样以避免大矩阵乘法
                if leaf_embs.size(0) > 1000:
                    indices = torch.randperm(leaf_embs.size(0))[:1000]
                    leaf_embs_norm = leaf_embs_norm[indices]
                
                leaf_similarity = torch.mm(leaf_embs_norm, leaf_embs_norm.t())
                loss_leaf = leaf_similarity.mean()
                total_loss += 0.1 * leaf_boost * loss_leaf
        
        return total_loss
    
    def _link_prediction_loss(self, embeddings, edge_index):
        """链接预测损失"""
        if edge_index.size(1) == 0:
            return torch.tensor(0.0, device=self.device)
        
        pos_score = (embeddings[edge_index[0]] * embeddings[edge_index[1]]).sum(dim=1)
        
        num_neg = min(edge_index.size(1), 1000)
        neg_edge_index = self._negative_sampling(edge_index, embeddings.size(0), num_neg)
        neg_score = (embeddings[neg_edge_index[0]] * embeddings[neg_edge_index[1]]).sum(dim=1)
        
        loss = -torch.log(torch.sigmoid(pos_score) + 1e-15).mean() - \
               torch.log(1 - torch.sigmoid(neg_score) + 1e-15).mean()
        
        return loss
    
    def _namespace_similarity_loss(self, ns_embeddings):
        """子本体内相似性损失"""
        ns_embeddings_norm = F.normalize(ns_embeddings, p=2, dim=1)
        
        if ns_embeddings.size(0) > 1000:
            indices = torch.randperm(ns_embeddings.size(0))[:1000]
            ns_embeddings_norm = ns_embeddings_norm[indices]
        
        similarity = torch.mm(ns_embeddings_norm, ns_embeddings_norm.t())
        return -similarity.mean()
    
    def _hierarchy_consistency_loss(self, embeddings, is_a_edges):
        """层次一致性损失"""
        if is_a_edges.size(1) == 0:
            return torch.tensor(0.0, device=self.device)
        
        child_emb = embeddings[is_a_edges[0]]
        parent_emb = embeddings[is_a_edges[1]]
        similarity = F.cosine_similarity(child_emb, parent_emb, dim=1)
        return -similarity.mean()
    
    def _negative_sampling(self, pos_edge_index, num_nodes, num_neg):
        """负采样"""
        neg_edges = []
        pos_edges_set = set(map(tuple, pos_edge_index.t().cpu().numpy()))
        
        max_attempts = num_neg * 10
        attempts = 0
        
        while len(neg_edges) < num_neg and attempts < max_attempts:
            src = torch.randint(0, num_nodes, (1,)).item()
            tgt = torch.randint(0, num_nodes, (1,)).item()
            if (src, tgt) not in pos_edges_set and src != tgt:
                neg_edges.append([src, tgt])
            attempts += 1
        
        if len(neg_edges) == 0:
            neg_edges = [[0, 1]]
        
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
            'num_epochs': 300,
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
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb)
        )
        
        top_indices = np.argsort(similarities)[::-1][1:6]
        
        idx_to_go_id = {v: k for k, v in self.go_id_to_idx.items()}
        
        print(f"\n  查询GO term: {query_go_id}")
        print(f"  最相似的5个GO terms:")
        for rank, idx in enumerate(top_indices, 1):
            go_id = idx_to_go_id[idx]
            sim = similarities[idx]
            print(f"    {rank}. {go_id} (相似度: {sim:.4f})")

# ==================== 9. 主程序 ====================
def main(obo_file_path, output_dir='./go_gnn_results', config_mode='balanced'):
    """主函数"""
    print("="*80)
    print("GO Term GNN - 完整训练流程")
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
        print("[步骤 1/6] 解析OBO文件")
        print("="*80)
        parser = OBOParser(obo_file_path)
        go_terms, relationships, namespace_terms = parser.parse()
        
        # 2. 构建图
        print("\n" + "="*80)
        print("[步骤 2/6] 构建层次感知图")
        print("="*80)
        builder = HierarchicalGOGraphBuilder(go_terms, relationships, namespace_terms)
        data, go_id_to_idx, idx_to_go_id = builder.build_hierarchical_graph()
        
        # 3. 创建模型
        print("\n" + "="*80)
        print("[步骤 3/6] 创建GNN模型")
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
        print("[步骤 4/6] 训练模型")
        print("="*80)
        
        trainer = OptimizedTrainer(model, data, config, device=device)
        embeddings = trainer.train()
        
        # 5. 保存结果
        print("\n" + "="*80)
        print("[步骤 5/6] 保存结果")
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
        print("[步骤 6/6] 分析结果")
        print("="*80)
        
        analyzer = EmbeddingAnalyzer(embeddings, data, go_id_to_idx, 
                                    namespace_terms, output_dir)
        analyzer.analyze()
        
        print("\n" + "="*80)
        print("训练完成!")
        print("="*80)
        print(f"\n输出文件:")
        print(f"  ├─ go_term_embeddings.npy  ({embeddings.shape[0]} x {embeddings.shape[1]})")
        print(f"  ├─ metadata.pkl")
        print(f"  ├─ training_history.png")
        print(f"  └─ embeddings_tsne.png")
        print(f"\n所有文件保存在: {output_dir}")
        print("="*80 + "\n")
        
        return embeddings, metadata
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

# ==================== 10. 工具函数 ====================
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
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb)
    )
    
    top_indices = np.argsort(similarities)[::-1][1:k+1]
    
    results = []
    for idx in top_indices:
        results.append({
            'go_id': idx_to_go_id[idx],
            'similarity': similarities[idx]
        })
    
    return results

# ==================== 11. 命令行接口 ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='GO Term GNN训练')
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
