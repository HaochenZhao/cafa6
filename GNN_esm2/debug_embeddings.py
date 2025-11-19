"""
调试ESM2 embeddings文件

检查npz文件中的实际内容和keys
"""

import numpy as np
import sys

def check_embeddings_file(emb_file):
    """检查embeddings文件内容"""
    
    print("="*70)
    print(f"检查文件: {emb_file}")
    print("="*70)
    
    try:
        data = np.load(emb_file)
        
        print(f"\n✓ 文件加载成功")
        print(f"\n可用的keys:")
        for i, key in enumerate(data.keys(), 1):
            print(f"  {i}. '{key}'")
        
        print(f"\n详细信息:")
        for key in data.keys():
            value = data[key]
            print(f"\n  Key: '{key}'")
            print(f"    Type: {type(value)}")
            
            if isinstance(value, np.ndarray):
                print(f"    Shape: {value.shape}")
                print(f"    Dtype: {value.dtype}")
                print(f"    Size: {value.nbytes / 1024 / 1024:.2f} MB")
                
                if len(value.shape) == 2:
                    print(f"    说明: 可能是 embeddings [num_samples, dim]")
                elif len(value.shape) == 1:
                    if value.dtype.kind in ['U', 'S', 'O']:
                        print(f"    说明: 可能是 IDs 或字符串数组")
                        print(f"    示例: {value[:3]}")
                    else:
                        print(f"    说明: 可能是 标量数组")
            elif isinstance(value, (str, int, float)):
                print(f"    Value: {value}")
        
        # 推断哪个是test embeddings
        print("\n" + "="*70)
        print("推断结果:")
        print("="*70)
        
        possible_test_keys = []
        
        for key in data.keys():
            value = data[key]
            if isinstance(value, np.ndarray) and len(value.shape) == 2:
                # 2D数组，可能是embeddings
                key_lower = key.lower()
                if 'test' in key_lower:
                    possible_test_keys.append((key, value.shape))
                    print(f"\n✓ 可能的test embeddings: '{key}'")
                    print(f"  Shape: {value.shape}")
        
        if not possible_test_keys:
            print("\n⚠️  未找到明显的test embeddings")
            print("\n所有2D数组:")
            for key in data.keys():
                value = data[key]
                if isinstance(value, np.ndarray) and len(value.shape) == 2:
                    print(f"  '{key}': {value.shape}")
        
        # 给出建议
        print("\n" + "="*70)
        print("建议:")
        print("="*70)
        
        if possible_test_keys:
            key, shape = possible_test_keys[0]
            print(f"\n使用 '{key}' 作为测试集embeddings")
            print(f"如果predict.py报错，请检查这个key名是否正确")
        else:
            print("\n请手动确认哪个key是测试集embeddings")
            print("通常key名包含 'test' 或 'testsuperset'")
        
        return True
        
    except FileNotFoundError:
        print(f"\n❌ 文件不存在: {emb_file}")
        return False
    except Exception as e:
        print(f"\n❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python GNN_esm2/debug_embeddings.py <embeddings_file>")
        print("\n示例:")
        print("  python GNN_esm2/debug_embeddings.py embeddings/embeddings_t6_8M_mean.npz")
        sys.exit(1)
    
    emb_file = sys.argv[1]
    check_embeddings_file(emb_file)