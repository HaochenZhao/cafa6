"""
ESM2 Embeddings预计算脚本 - 兼容processed_data_v4
从cafa主目录运行: python GNN_esm2/precompute_embeddings.py

关键特性:
1. 从原始FASTA文件加载序列（processed_data_v4不含序列）
2. 按照processed_data_v4的划分生成embeddings
3. 支持多种ESM2模型
4. 自动批处理和GPU加速
"""

import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
from Bio import SeqIO
from typing import Dict, List
import gc


def load_esm2_model(model_name: str, device: torch.device):
    """
    加载ESM2模型
    
    支持的模型:
    - esm2_t6_8M_UR50D (320维)
    - esm2_t12_35M_UR50D (480维)
    - esm2_t30_150M_UR50D (640维)
    - esm2_t33_650M_UR50D (1280维)
    """
    print(f"\n加载ESM2模型: {model_name}")
    
    import esm
    
    if model_name == "esm2_t6_8M_UR50D":
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        emb_dim = 320
    elif model_name == "esm2_t12_35M_UR50D":
        model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        emb_dim = 480
    elif model_name == "esm2_t30_150M_UR50D":
        model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        emb_dim = 640
    elif model_name == "esm2_t33_650M_UR50D":
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        emb_dim = 1280
    else:
        raise ValueError(f"不支持的模型: {model_name}")
    
    model = model.to(device)
    model.eval()
    
    batch_converter = alphabet.get_batch_converter()
    
    print(f"✓ 模型加载完成")
    print(f"  Embedding维度: {emb_dim}")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, batch_converter, emb_dim


def load_sequences_from_fasta(fasta_file: str, protein_ids: np.ndarray) -> Dict[str, str]:
    """
    从FASTA文件加载指定蛋白质的序列
    
    参数:
        fasta_file: FASTA文件路径
        protein_ids: 需要加载的蛋白质ID数组
    
    返回:
        {protein_id: sequence}
    """
    print(f"\n从FASTA加载序列: {fasta_file}")
    
    protein_ids_set = set(protein_ids)
    sequences = {}
    
    for record in tqdm(SeqIO.parse(fasta_file, "fasta"), desc="读取FASTA"):
        # 从 'sp|A0A0C5B5G6|MOTSC_HUMAN' 提取 'A0A0C5B5G6'
        if '|' in record.id:
            uniprot_id = record.id.split('|')[1]
        else:
            uniprot_id = record.id
        
        if uniprot_id in protein_ids_set:
            sequences[uniprot_id] = str(record.seq)
    
    print(f"✓ 加载序列: {len(sequences)} / {len(protein_ids)} 条")
    
    if len(sequences) < len(protein_ids):
        missing = set(protein_ids) - set(sequences.keys())
        print(f"⚠️  警告: {len(missing)} 个蛋白质未找到序列")
        if len(missing) <= 10:
            print(f"  缺失ID: {list(missing)}")
    
    return sequences


def compute_embeddings_batch(
    model,
    batch_converter,
    sequences_dict: Dict[str, str],
    protein_ids: np.ndarray,
    device: torch.device,
    batch_size: int = 8,
    pooling: str = "mean"
) -> np.ndarray:
    """
    批量计算ESM2 embeddings
    
    参数:
        pooling: mean (平均池化) 或 cls (使用CLS token)
    
    返回:
        embeddings: [num_proteins, emb_dim]
    """
    print(f"\n计算ESM2 embeddings:")
    print(f"  蛋白质数: {len(protein_ids)}")
    print(f"  批次大小: {batch_size}")
    print(f"  池化方式: {pooling}")
    
    all_embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(protein_ids), batch_size), desc="处理批次"):
            batch_ids = protein_ids[i:i+batch_size]
            
            # 准备batch数据
            batch_data = []
            for protein_id in batch_ids:
                if protein_id in sequences_dict:
                    seq = sequences_dict[protein_id]
                    # 截断过长序列
                    if len(seq) > 1024:
                        seq = seq[:1024]
                    batch_data.append((protein_id, seq))
                else:
                    # 缺失序列用空序列占位
                    batch_data.append((protein_id, "M"))  # 最短的有效序列
            
            # 转换为模型输入
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
            batch_tokens = batch_tokens.to(device)
            
            # 前向传播
            results = model(batch_tokens, repr_layers=[model.num_layers])
            token_representations = results["representations"][model.num_layers]
            
            # 池化
            for j, (protein_id, seq) in enumerate(batch_data):
                seq_len = len(seq)
                
                if pooling == "mean":
                    # 平均池化（不包括特殊token）
                    emb = token_representations[j, 1:seq_len+1].mean(0)
                elif pooling == "cls":
                    # 使用CLS token
                    emb = token_representations[j, 0]
                else:
                    raise ValueError(f"不支持的池化方式: {pooling}")
                
                all_embeddings.append(emb.cpu().numpy())
            
            # 清理GPU内存
            del batch_tokens, results, token_representations
            if i % 10 == 0:
                torch.cuda.empty_cache()
    
    embeddings = np.stack(all_embeddings)
    print(f"✓ Embeddings计算完成: {embeddings.shape}")
    
    return embeddings


def main():
    parser = argparse.ArgumentParser(description="ESM2 Embeddings预计算")
    parser.add_argument('--model_name', type=str, 
                        default='esm2_t6_8M_UR50D',
                        choices=['esm2_t6_8M_UR50D', 'esm2_t12_35M_UR50D', 
                                'esm2_t30_150M_UR50D', 'esm2_t33_650M_UR50D'],
                        help='ESM2模型名称')
    parser.add_argument('--processed_data_dir', type=str,
                        default='processed_data_v4',
                        help='processed数据目录')
    parser.add_argument('--fasta_dir', type=str,
                        default='cafa-6-protein-function-prediction',
                        help='原始FASTA文件目录')
    parser.add_argument('--output_dir', type=str,
                        default='embeddings',
                        help='输出目录')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批次大小')
    parser.add_argument('--pooling', type=str, default='mean',
                        choices=['mean', 'cls'],
                        help='池化方式')
    parser.add_argument('--device', type=str, default='auto',
                        help='设备 (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # 确保在cafa主目录下运行
    if not os.path.exists(args.processed_data_dir):
        print(f"错误: 找不到processed数据目录 {args.processed_data_dir}")
        print("请先运行: python GNN_esm2/preprocess_data.py")
        sys.exit(1)
    
    print("="*70)
    print("ESM2 Embeddings预计算")
    print("="*70)
    print(f"\n配置:")
    print(f"  模型: {args.model_name}")
    print(f"  Processed数据: {args.processed_data_dir}")
    print(f"  FASTA目录: {args.fasta_dir}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  池化方式: {args.pooling}")
    
    # 设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"  设备: {device}")
    
    # 加载processed数据（只需要protein IDs）
    print("\n" + "="*70)
    print("加载processed数据")
    print("="*70)
    
    data_file = os.path.join(args.processed_data_dir, 'cafa6_data.npz')
    data = np.load(data_file, allow_pickle=True)
    
    train_ids = data['train_ids']
    val_ids = data['val_ids']
    test_ids = data.get('test_ids', None)
    
    print(f"✓ 训练集: {len(train_ids):,} 蛋白质")
    print(f"✓ 验证集: {len(val_ids):,} 蛋白质")
    if test_ids is not None:
        print(f"✓ 测试集: {len(test_ids):,} 蛋白质")
    
    # 加载ESM2模型
    print("\n" + "="*70)
    print("加载ESM2模型")
    print("="*70)
    
    model, batch_converter, emb_dim = load_esm2_model(args.model_name, device)
    
    # FASTA文件路径
    train_fasta = os.path.join(args.fasta_dir, 'Train', 'train_sequences.fasta')
    test_fasta = os.path.join(args.fasta_dir, 'testsuperset.fasta')
    
    # 加载训练集和验证集序列
    print("\n" + "="*70)
    print("加载训练集和验证集序列")
    print("="*70)
    
    all_train_val_ids = np.concatenate([train_ids, val_ids])
    train_val_sequences = load_sequences_from_fasta(train_fasta, all_train_val_ids)
    
    # 计算训练集embeddings
    print("\n" + "="*70)
    print("计算训练集embeddings")
    print("="*70)
    
    train_embeddings = compute_embeddings_batch(
        model, batch_converter, train_val_sequences, train_ids,
        device, args.batch_size, args.pooling
    )
    
    # 清理内存
    torch.cuda.empty_cache()
    gc.collect()
    
    # 计算验证集embeddings
    print("\n" + "="*70)
    print("计算验证集embeddings")
    print("="*70)
    
    val_embeddings = compute_embeddings_batch(
        model, batch_converter, train_val_sequences, val_ids,
        device, args.batch_size, args.pooling
    )
    
    # 清理
    del train_val_sequences
    torch.cuda.empty_cache()
    gc.collect()
    
    # 计算测试集embeddings（如果存在）
    test_embeddings = None
    if test_ids is not None and os.path.exists(test_fasta):
        print("\n" + "="*70)
        print("计算测试集embeddings")
        print("="*70)
        
        test_sequences = load_sequences_from_fasta(test_fasta, test_ids)
        
        test_embeddings = compute_embeddings_batch(
            model, batch_converter, test_sequences, test_ids,
            device, args.batch_size, args.pooling
        )
        
        del test_sequences
        torch.cuda.empty_cache()
        gc.collect()
    
    # 保存embeddings
    print("\n" + "="*70)
    print("保存embeddings")
    print("="*70)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 文件名包含模型名和池化方式
    model_short = args.model_name.replace('esm2_', '').replace('_UR50D', '')
    output_file = os.path.join(
        args.output_dir, 
        f"embeddings_{model_short}_{args.pooling}.npz"
    )
    
    save_dict = {
        'train_embeddings': train_embeddings,
        'val_embeddings': val_embeddings,
        'model_name': args.model_name,
        'pooling': args.pooling,
        'embedding_dim': emb_dim
    }
    
    if test_embeddings is not None:
        save_dict['test_embeddings'] = test_embeddings
    
    np.savez_compressed(output_file, **save_dict)
    
    print(f"✓ 保存到: {output_file}")
    
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✓ 文件大小: {file_size_mb:.2f} MB")
    
    # 保存元数据
    import json
    
    metadata = {
        'model_name': args.model_name,
        'pooling': args.pooling,
        'embedding_dim': int(emb_dim),
        'num_train': int(len(train_ids)),
        'num_val': int(len(val_ids)),
        'num_test': int(len(test_ids)) if test_ids is not None else 0,
        'batch_size': args.batch_size
    }
    
    metadata_file = os.path.join(
        args.output_dir,
        f"embeddings_{model_short}_{args.pooling}_metadata.json"
    )
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ 元数据保存到: {metadata_file}")
    
    # 打印摘要
    print("\n" + "="*70)
    print("Embeddings摘要")
    print("="*70)
    print(f"模型: {args.model_name}")
    print(f"Embedding维度: {emb_dim}")
    print(f"池化方式: {args.pooling}")
    print(f"\n训练集: {train_embeddings.shape}")
    print(f"验证集: {val_embeddings.shape}")
    if test_embeddings is not None:
        print(f"测试集: {test_embeddings.shape}")
    
    print("\n" + "="*70)
    print("✅ Embeddings预计算完成！")
    print("="*70)
    print(f"\n下一步:")
    print(f"  1. 更新config.yaml中的embeddings_file路径:")
    print(f"     esm2:")
    print(f"       embeddings_file: \"{output_file}\"")
    print(f"  2. 运行测试: python GNN_esm2/test_all.py")
    print(f"  3. 开始训练: python GNN_esm2/train.py")


if __name__ == "__main__":
    main()