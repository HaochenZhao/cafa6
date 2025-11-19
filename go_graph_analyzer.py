import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter, deque
import networkx as nx
from typing import Dict, List, Tuple, Set
import pickle
import json
from datetime import datetime

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
        
        # 验证文件存在
        if not os.path.exists(obo_file_path):
            raise FileNotFoundError(f"OBO file not found: {obo_file_path}")
    
    def parse(self) -> Tuple[Dict, List, Dict]:
        """
        解析OBO文件
        Returns:
            go_terms: {go_id: term_info}
            relationships: [(source, target, relation_type)]
            namespace_terms: {namespace: set(go_ids)}
        """
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
            
            # 处理最后一个term
            if current_term:
                self._process_term(current_term)
        
        print(f"解析完成! 共处理 {line_count:,} 行")
        print(f"提取到 {len(self.go_terms):,} 个有效GO terms")
        print(f"提取到 {len(self.relationships):,} 条关系")
        
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
            # 提取定义文本（去除引号）
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
                
        elif line.startswith("alt_id:"):
            alt_id = line.split("alt_id:")[1].strip()
            if 'alt_ids' not in current_term:
                current_term['alt_ids'] = []
            current_term['alt_ids'].append(alt_id)
    
    def _process_term(self, term: dict):
        """处理并存储一个GO term"""
        if 'id' not in term or term.get('is_obsolete', False):
            return
        
        go_id = term['id']
        self.go_terms[go_id] = term
        
        # 记录namespace
        if 'namespace' in term:
            ns = term['namespace']
            if ns in self.namespace_terms:
                self.namespace_terms[ns].add(go_id)
        
        # 添加is_a关系
        if 'is_a' in term:
            for parent in term['is_a']:
                self.relationships.append((go_id, parent, 'is_a'))
        
        # 添加其他关系
        if 'relationship' in term:
            for rel_type, target in term['relationship']:
                self.relationships.append((go_id, target, rel_type))

# ==================== 2. 图结构构建器 ====================
class GOGraphBuilder:
    """构建NetworkX图用于分析"""
    
    def __init__(self, go_terms: Dict, relationships: List):
        self.go_terms = go_terms
        self.relationships = relationships
        self.graph = nx.DiGraph()
        self.undirected_graph = None
        
    def build_graph(self):
        """构建有向图"""
        print("\n构建NetworkX图...")
        
        # 添加节点
        for go_id, term_info in self.go_terms.items():
            self.graph.add_node(
                go_id,
                name=term_info.get('name', ''),
                namespace=term_info.get('namespace', ''),
                definition=term_info.get('def', '')
            )
        
        # 添加边
        relation_types = Counter()
        for source, target, rel_type in self.relationships:
            if source in self.go_terms and target in self.go_terms:
                self.graph.add_edge(source, target, relation=rel_type)
                relation_types[rel_type] += 1
        
        # 创建无向图版本（用于某些分析）
        self.undirected_graph = self.graph.to_undirected()
        
        print(f"图构建完成:")
        print(f"  节点数: {self.graph.number_of_nodes():,}")
        print(f"  边数: {self.graph.number_of_edges():,}")
        print(f"  关系类型: {len(relation_types)}")
        for rel_type, count in relation_types.most_common():
            print(f"    - {rel_type}: {count:,}")
        
        return self.graph

# ==================== 3. 全面的图结构分析器 ====================
class ComprehensiveGOGraphAnalyzer:
    """全面分析GO图的结构特性"""
    
    def __init__(self, graph: nx.DiGraph, go_terms: Dict, namespace_terms: Dict):
        self.graph = graph
        self.go_terms = go_terms
        self.namespace_terms = namespace_terms
        self.undirected_graph = graph.to_undirected()
        
        # 存储分析结果
        self.analysis_results = {}
    
    def run_full_analysis(self, output_dir: str = "./go_analysis_results"):
        """运行完整的分析流程"""
        print("\n" + "="*80)
        print("开始全面的GO图结构分析")
        print("="*80)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        
        # 1. 基本统计
        print("\n[1/10] 基本统计信息...")
        self._analyze_basic_stats()
        
        # 2. 度分布
        print("\n[2/10] 度分布分析...")
        self._analyze_degree_distribution()
        
        # 3. 连通性
        print("\n[3/10] 连通性分析...")
        self._analyze_connectivity()
        
        # 4. 层次结构
        print("\n[4/10] 层次结构分析...")
        self._analyze_hierarchy()
        
        # 5. 子本体分析
        print("\n[5/10] 子本体（命名空间）分析...")
        self._analyze_namespaces()
        
        # 6. 路径分析
        print("\n[6/10] 路径长度分析...")
        self._analyze_path_lengths()
        
        # 7. 中心性分析
        print("\n[7/10] 节点中心性分析...")
        self._analyze_centrality()
        
        # 8. 社区结构
        print("\n[8/10] 社区结构分析...")
        self._analyze_community_structure()
        
        # 9. 关系类型分析
        print("\n[9/10] 关系类型分析...")
        self._analyze_relation_types()
        
        # 10. 生成报告
        print("\n[10/10] 生成分析报告...")
        self._generate_report()
        
        print("\n" + "="*80)
        print(f"分析完成! 结果保存在: {output_dir}")
        print("="*80)
        
        return self.analysis_results
    
    def _analyze_basic_stats(self):
        """基本统计信息"""
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'num_namespaces': len(self.namespace_terms),
            'density': nx.density(self.graph),
            'is_dag': nx.is_directed_acyclic_graph(self.graph)
        }
        
        print(f"  节点数量: {stats['num_nodes']:,}")
        print(f"  边数量: {stats['num_edges']:,}")
        print(f"  图密度: {stats['density']:.6f}")
        print(f"  是否为DAG: {'是' if stats['is_dag'] else '否'}")
        
        # 统计节点和边属性
        stats['nodes_with_names'] = sum(1 for n in self.graph.nodes() 
                                       if self.graph.nodes[n].get('name'))
        stats['nodes_with_def'] = sum(1 for n in self.graph.nodes() 
                                      if self.graph.nodes[n].get('definition'))
        
        print(f"  有名称的节点: {stats['nodes_with_names']:,}")
        print(f"  有定义的节点: {stats['nodes_with_def']:,}")
        
        self.analysis_results['basic_stats'] = stats
    
    def _analyze_degree_distribution(self):
        """度分布分析"""
        in_degrees = [d for n, d in self.graph.in_degree()]
        out_degrees = [d for n, d in self.graph.out_degree()]
        total_degrees = [d for n, d in self.undirected_graph.degree()]
        
        degree_stats = {
            'in_degree': {
                'mean': np.mean(in_degrees),
                'median': np.median(in_degrees),
                'max': np.max(in_degrees),
                'min': np.min(in_degrees),
                'std': np.std(in_degrees)
            },
            'out_degree': {
                'mean': np.mean(out_degrees),
                'median': np.median(out_degrees),
                'max': np.max(out_degrees),
                'min': np.min(out_degrees),
                'std': np.std(out_degrees)
            },
            'total_degree': {
                'mean': np.mean(total_degrees),
                'median': np.median(total_degrees),
                'max': np.max(total_degrees),
                'std': np.std(total_degrees)
            }
        }
        
        print(f"  入度 - 平均: {degree_stats['in_degree']['mean']:.2f}, "
              f"最大: {degree_stats['in_degree']['max']}")
        print(f"  出度 - 平均: {degree_stats['out_degree']['mean']:.2f}, "
              f"最大: {degree_stats['out_degree']['max']}")
        
        # 找到高度节点
        high_in_degree_nodes = sorted(self.graph.in_degree(), key=lambda x: x[1], reverse=True)[:10]
        high_out_degree_nodes = sorted(self.graph.out_degree(), key=lambda x: x[1], reverse=True)[:10]
        
        degree_stats['top_in_degree_nodes'] = [
            (node, degree, self.graph.nodes[node].get('name', 'N/A'))
            for node, degree in high_in_degree_nodes
        ]
        degree_stats['top_out_degree_nodes'] = [
            (node, degree, self.graph.nodes[node].get('name', 'N/A'))
            for node, degree in high_out_degree_nodes
        ]
        
        print(f"\n  Top 3 入度最高的节点:")
        for node, degree, name in degree_stats['top_in_degree_nodes'][:3]:
            print(f"    - {node} ({name[:50]}...): {degree}")
        
        # 绘制度分布图
        self._plot_degree_distribution(in_degrees, out_degrees, total_degrees)
        
        self.analysis_results['degree_distribution'] = degree_stats
    
    def _analyze_connectivity(self):
        """连通性分析"""
        connectivity = {}
        
        # 弱连通分量
        weak_components = list(nx.weakly_connected_components(self.graph))
        connectivity['num_weak_components'] = len(weak_components)
        connectivity['largest_weak_component_size'] = len(max(weak_components, key=len))
        
        # 强连通分量
        strong_components = list(nx.strongly_connected_components(self.graph))
        connectivity['num_strong_components'] = len(strong_components)
        connectivity['largest_strong_component_size'] = len(max(strong_components, key=len))
        
        print(f"  弱连通分量数: {connectivity['num_weak_components']}")
        print(f"  最大弱连通分量: {connectivity['largest_weak_component_size']:,} 节点")
        print(f"  强连通分量数: {connectivity['num_strong_components']}")
        
        self.analysis_results['connectivity'] = connectivity
    
    def _analyze_hierarchy(self):
        """层次结构分析"""
        print("  正在计算节点深度...")
        
        # 找到所有根节点（出度为0，即没有父节点）
        roots = [n for n in self.graph.nodes() if self.graph.out_degree(n) == 0]
        
        # 找到所有叶子节点（入度为0，即没有子节点）
        leaves = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]
        
        hierarchy = {
            'num_roots': len(roots),
            'num_leaves': len(leaves),
            'root_nodes': roots[:10]  # 保存前10个根节点
        }
        
        print(f"  根节点数: {len(roots)}")
        print(f"  叶子节点数: {len(leaves)}")
        
        # 计算深度（到最近根节点的最短路径）
        depths = []
        depth_dict = {}
        
        for node in self.graph.nodes():
            min_depth = float('inf')
            for root in roots:
                try:
                    if nx.has_path(self.graph, node, root):
                        depth = nx.shortest_path_length(self.graph, node, root)
                        min_depth = min(min_depth, depth)
                except nx.NetworkXError:
                    continue
            
            if min_depth != float('inf'):
                depths.append(min_depth)
                depth_dict[node] = min_depth
            else:
                depth_dict[node] = 0
        
        if depths:
            hierarchy['max_depth'] = int(np.max(depths))
            hierarchy['mean_depth'] = float(np.mean(depths))
            hierarchy['median_depth'] = float(np.median(depths))
            
            print(f"  最大深度: {hierarchy['max_depth']}")
            print(f"  平均深度: {hierarchy['mean_depth']:.2f}")
            print(f"  中位深度: {hierarchy['median_depth']:.0f}")
            
            # 深度分布
            depth_distribution = Counter(depths)
            hierarchy['depth_distribution'] = dict(depth_distribution)
            
            # 按深度分组节点
            level_groups = defaultdict(list)
            for node, depth in depth_dict.items():
                level_groups[depth].append(node)
            
            hierarchy['nodes_by_level'] = {k: len(v) for k, v in level_groups.items()}
            
            print(f"\n  各层次节点分布:")
            for level in sorted(level_groups.keys())[:10]:  # 只显示前10层
                count = len(level_groups[level])
                print(f"    深度 {level}: {count:,} 个节点")
            
            # 绘制深度分布
            self._plot_depth_distribution(depth_distribution)
        
        self.analysis_results['hierarchy'] = hierarchy
    
    def _analyze_namespaces(self):
        """子本体（命名空间）分析"""
        namespace_analysis = {}
        
        for namespace, go_ids in self.namespace_terms.items():
            if not go_ids:
                continue
            
            # 创建子图
            subgraph = self.graph.subgraph(go_ids)
            
            ns_stats = {
                'num_nodes': subgraph.number_of_nodes(),
                'num_edges': subgraph.number_of_edges(),
                'density': nx.density(subgraph) if subgraph.number_of_nodes() > 0 else 0,
                'avg_in_degree': np.mean([d for n, d in subgraph.in_degree()]) if subgraph.number_of_nodes() > 0 else 0,
                'avg_out_degree': np.mean([d for n, d in subgraph.out_degree()]) if subgraph.number_of_nodes() > 0 else 0
            }
            
            namespace_analysis[namespace] = ns_stats
            
            print(f"\n  {namespace}:")
            print(f"    节点数: {ns_stats['num_nodes']:,}")
            print(f"    边数: {ns_stats['num_edges']:,}")
            print(f"    密度: {ns_stats['density']:.6f}")
        
        self.analysis_results['namespace_analysis'] = namespace_analysis
    
    def _analyze_path_lengths(self):
        """路径长度分析（采样）"""
        print("  使用采样方法估计路径长度...")
        
        # 采样节点
        nodes = list(self.graph.nodes())
        sample_size = min(1000, len(nodes))
        sampled_nodes = np.random.choice(nodes, sample_size, replace=False)
        
        path_lengths = []
        
        # 计算采样节点间的路径
        for i, source in enumerate(sampled_nodes[:200]):  # 限制源节点数量
            for target in sampled_nodes[i+1:i+6]:  # 每个源节点只测试5个目标
                try:
                    if nx.has_path(self.undirected_graph, source, target):
                        length = nx.shortest_path_length(self.undirected_graph, source, target)
                        path_lengths.append(length)
                except nx.NetworkXError:
                    continue
        
        if path_lengths:
            path_stats = {
                'mean': np.mean(path_lengths),
                'median': np.median(path_lengths),
                'max': np.max(path_lengths),
                'min': np.min(path_lengths),
                'percentile_90': np.percentile(path_lengths, 90),
                'percentile_95': np.percentile(path_lengths, 95)
            }
            
            print(f"  平均路径长度: {path_stats['mean']:.2f}")
            print(f"  中位路径长度: {path_stats['median']:.0f}")
            print(f"  90分位路径长度: {path_stats['percentile_90']:.0f}")
            print(f"  最大路径长度: {path_stats['max']}")
            
            # 估计图直径
            path_stats['estimated_diameter'] = path_stats['percentile_95']
            print(f"  估计直径 (95分位): {path_stats['estimated_diameter']:.0f}")
            
            self.analysis_results['path_lengths'] = path_stats
        else:
            print("  警告: 无法计算路径长度")
            self.analysis_results['path_lengths'] = None
    
    def _analyze_centrality(self):
        """节点中心性分析"""
        print("  计算节点中心性（这可能需要一些时间）...")
        
        centrality_results = {}
        
        # 1. 度中心性（快速）
        degree_centrality = nx.degree_centrality(self.undirected_graph)
        top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        centrality_results['degree_centrality_top10'] = [
            (node, score, self.graph.nodes[node].get('name', 'N/A'))
            for node, score in top_degree
        ]
        
        print(f"  Top 3 度中心性节点:")
        for node, score, name in centrality_results['degree_centrality_top10'][:3]:
            print(f"    - {node} ({name[:50]}...): {score:.4f}")
        
        # 2. 介数中心性（采样，因为计算昂贵）
        if self.graph.number_of_nodes() < 5000:
            print("  计算介数中心性...")
            betweenness = nx.betweenness_centrality(self.graph, k=min(500, self.graph.number_of_nodes()))
            top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
            centrality_results['betweenness_centrality_top10'] = [
                (node, score, self.graph.nodes[node].get('name', 'N/A'))
                for node, score in top_betweenness
            ]
        else:
            print("  跳过介数中心性计算（图太大）")
        
        # 3. PageRank
        print("  计算PageRank...")
        pagerank = nx.pagerank(self.graph, alpha=0.85)
        top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
        centrality_results['pagerank_top10'] = [
            (node, score, self.graph.nodes[node].get('name', 'N/A'))
            for node, score in top_pagerank
        ]
        
        print(f"  Top 3 PageRank节点:")
        for node, score, name in centrality_results['pagerank_top10'][:3]:
            print(f"    - {node} ({name[:50]}...): {score:.6f}")
        
        self.analysis_results['centrality'] = centrality_results
    
    def _analyze_community_structure(self):
        """社区结构分析"""
        print("  检测社区结构...")
        
        # 使用Louvain算法（需要python-louvain包）
        try:
            import community as community_louvain
            
            # 转换为无向图并使用Louvain
            partition = community_louvain.best_partition(self.undirected_graph)
            
            num_communities = len(set(partition.values()))
            community_sizes = Counter(partition.values())
            
            community_info = {
                'num_communities': num_communities,
                'largest_community_size': max(community_sizes.values()),
                'smallest_community_size': min(community_sizes.values()),
                'avg_community_size': np.mean(list(community_sizes.values())),
                'modularity': community_louvain.modularity(partition, self.undirected_graph)
            }
            
            print(f"  检测到 {num_communities} 个社区")
            print(f"  最大社区: {community_info['largest_community_size']:,} 节点")
            print(f"  模块度: {community_info['modularity']:.4f}")
            
            self.analysis_results['community_structure'] = community_info
            
        except ImportError:
            print("  警告: 未安装python-louvain包，跳过社区检测")
            print("  安装方法: pip install python-louvain")
            self.analysis_results['community_structure'] = None
    
    def _analyze_relation_types(self):
        """关系类型详细分析"""
        relation_stats = defaultdict(lambda: {'count': 0, 'sources': set(), 'targets': set()})
        
        for source, target in self.graph.edges():
            rel_type = self.graph[source][target].get('relation', 'unknown')
            relation_stats[rel_type]['count'] += 1
            relation_stats[rel_type]['sources'].add(source)
            relation_stats[rel_type]['targets'].add(target)
        
        relation_analysis = {}
        for rel_type, stats in relation_stats.items():
            relation_analysis[rel_type] = {
                'count': stats['count'],
                'num_unique_sources': len(stats['sources']),
                'num_unique_targets': len(stats['targets'])
            }
            
            print(f"  {rel_type}:")
            print(f"    边数: {stats['count']:,}")
            print(f"    涉及 {len(stats['sources']):,} 个源节点, {len(stats['targets']):,} 个目标节点")
        
        self.analysis_results['relation_types'] = relation_analysis
    
    def _plot_degree_distribution(self, in_degrees, out_degrees, total_degrees):
        """绘制度分布图"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # 入度分布
        axes[0].hist(in_degrees, bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('In-Degree')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('In-Degree Distribution')
        axes[0].set_yscale('log')
        
        # 出度分布
        axes[1].hist(out_degrees, bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[1].set_xlabel('Out-Degree')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Out-Degree Distribution')
        axes[1].set_yscale('log')
        
        # 总度数分布
        axes[2].hist(total_degrees, bins=50, edgecolor='black', alpha=0.7, color='green')
        axes[2].set_xlabel('Total Degree')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Total Degree Distribution')
        axes[2].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'degree_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  度分布图已保存")
    
    def _plot_depth_distribution(self, depth_distribution):
        """绘制深度分布图"""
        depths = sorted(depth_distribution.keys())
        counts = [depth_distribution[d] for d in depths]
        
        plt.figure(figsize=(12, 6))
        plt.bar(depths, counts, edgecolor='black', alpha=0.7)
        plt.xlabel('Depth (Distance to Root)')
        plt.ylabel('Number of Nodes')
        plt.title('Hierarchical Depth Distribution')
        plt.yscale('log')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'depth_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  深度分布图已保存")
    
    def _generate_report(self):
        """生成分析报告"""
        report_path = os.path.join(self.output_dir, 'analysis_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("GO图结构分析报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            # 基本统计
            f.write("[1] 基本统计\n")
            f.write("-"*80 + "\n")
            for key, value in self.analysis_results['basic_stats'].items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # 度分布
            f.write("[2] 度分布\n")
            f.write("-"*80 + "\n")
            for degree_type in ['in_degree', 'out_degree', 'total_degree']:
                f.write(f"\n{degree_type}:\n")
                for stat_name, stat_value in self.analysis_results['degree_distribution'][degree_type].items():
                    f.write(f"  {stat_name}: {stat_value}\n")
            f.write("\n")
            
            # 层次结构
            f.write("[3] 层次结构\n")
            f.write("-"*80 + "\n")
            for key, value in self.analysis_results['hierarchy'].items():
                if key not in ['depth_distribution', 'nodes_by_level', 'root_nodes']:
                    f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # 命名空间
            f.write("[4] 命名空间分析\n")
            f.write("-"*80 + "\n")
            for ns, stats in self.analysis_results['namespace_analysis'].items():
                f.write(f"\n{ns}:\n")
                for key, value in stats.items():
                    f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            # GNN超参数推荐
            f.write("[5] GNN超参数推荐\n")
            f.write("-"*80 + "\n")
            self._add_hyperparameter_recommendations(f)
        
        print(f"  分析报告已保存: {report_path}")
        
        # 保存JSON格式
        json_path = os.path.join(self.output_dir, 'analysis_results.json')
        
        # 转换不可序列化的对象
        json_results = {}
        for key, value in self.analysis_results.items():
            if isinstance(value, dict):
                json_results[key] = {k: (v if not isinstance(v, (set, np.integer, np.floating)) else 
                                        (list(v) if isinstance(v, set) else float(v) if isinstance(v, (np.integer, np.floating)) else v))
                                    for k, v in value.items()}
            else:
                json_results[key] = value
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"  JSON结果已保存: {json_path}")
    
    def _add_hyperparameter_recommendations(self, file_handle):
        """添加超参数推荐到报告"""
        stats = self.analysis_results
        
        # 基于分析结果推荐超参数
        num_nodes = stats['basic_stats']['num_nodes']
        max_depth = stats['hierarchy'].get('max_depth', 20)
        avg_path = stats['path_lengths']['mean'] if stats.get('path_lengths') else 10
        
        file_handle.write(f"基于图结构分析的GNN超参数推荐:\n\n")
        
        # 网络层数
        if max_depth <= 10:
            rec_layers = 3
        elif max_depth <= 20:
            rec_layers = 5
        else:
            rec_layers = min(7, max_depth // 3)
        
        file_handle.write(f"推荐网络层数: {rec_layers}\n")
        file_handle.write(f"  理由: 图最大深度为{max_depth}，{rec_layers}层足以覆盖大部分路径\n\n")
        
        # 隐藏维度
        if num_nodes < 1000:
            rec_hidden = 256
        elif num_nodes < 10000:
            rec_hidden = 512
        else:
            rec_hidden = 512
        
        file_handle.write(f"推荐隐藏层维度: {rec_hidden}\n")
        file_handle.write(f"  理由: 节点数为{num_nodes:,}，{rec_hidden}维提供足够表达能力\n\n")
        
        # 输出维度
        rec_output = 128 if num_nodes < 5000 else 256
        file_handle.write(f"推荐输出维度: {rec_output}\n\n")
        
        # 其他建议
        file_handle.write("其他建议:\n")
        file_handle.write(f"  - Dropout: 0.4\n")
        file_handle.write(f"  - Learning rate: 0.005\n")
        file_handle.write(f"  - Epochs: 300\n")
        file_handle.write(f"  - 使用注意力机制: 是 (GAT)\n")
        file_handle.write(f"  - 注意力头数: 4\n")

# ==================== 4. 主程序 ====================
def main(obo_file_path: str, output_dir: str = "./go_analysis_results"):
    """
    主函数：从OBO文件开始的完整分析流程
    
    Args:
        obo_file_path: OBO文件路径
        output_dir: 输出目录路径
    """
    print("="*80)
    print("GO图结构完整分析程序")
    print("="*80)
    print(f"\n输入文件: {obo_file_path}")
    print(f"输出目录: {output_dir}\n")
    
    try:
        # 1. 解析OBO文件
        parser = OBOParser(obo_file_path)
        go_terms, relationships, namespace_terms = parser.parse()
        
        # 2. 构建图
        builder = GOGraphBuilder(go_terms, relationships)
        graph = builder.build_graph()
        
        # 3. 运行完整分析
        analyzer = ComprehensiveGOGraphAnalyzer(graph, go_terms, namespace_terms)
        results = analyzer.run_full_analysis(output_dir)
        
        # 4. 输出关键发现
        print("\n" + "="*80)
        print("关键发现总结")
        print("="*80)
        
        print(f"\n图规模: {results['basic_stats']['num_nodes']:,} 节点, "
              f"{results['basic_stats']['num_edges']:,} 边")
        
        print(f"\n层次深度: 最大 {results['hierarchy']['max_depth']}, "
              f"平均 {results['hierarchy']['mean_depth']:.2f}")
        
        if results.get('path_lengths'):
            print(f"\n路径长度: 平均 {results['path_lengths']['mean']:.2f}, "
                  f"估计直径 {results['path_lengths']['estimated_diameter']:.0f}")
        
        print(f"\n子本体分布:")
        for ns, stats in results['namespace_analysis'].items():
            print(f"  {ns}: {stats['num_nodes']:,} 节点")
        
        print(f"\n推荐GNN配置:")
        print(f"  - 网络层数: 5")
        print(f"  - 隐藏维度: 512")
        print(f"  - 输出维度: 256")
        
        print(f"\n所有结果已保存到: {output_dir}")
        print("="*80)
        
        return results
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ==================== 5. 使用示例 ====================
if __name__ == "__main__":
    # import argparse
    
    # # 命令行参数解析
    # parser = argparse.ArgumentParser(description='GO图结构分析工具')
    # parser.add_argument('--obo', type=str, required=True,
    #                    help='OBO文件路径 (例如: go-basic.obo)')
    # parser.add_argument('--output', type=str, default='./go_analysis_results',
    #                    help='输出目录路径 (默认: ./go_analysis_results)')
    
    # args = parser.parse_args()
    
    # # 运行分析
    # results = main(args.obo, args.output)
    
    # 如果你想直接在代码中运行（不使用命令行参数）:
    results = main('/home/h/haochenz/cafa/cafa-6-protein-function-prediction/Train/go-basic.obo', '/home/h/haochenz/cafa/go_analysis_results')