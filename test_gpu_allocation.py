#!/usr/bin/env python3
"""
测试GPU分配和分布式训练设置
"""

import torch
import os
import sys

def test_gpu_allocation():
    """测试GPU分配"""
    print("=" * 50)
    print("GPU分配测试")
    print("=" * 50)
    
    # 检查CUDA
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return
    
    # 检查GPU数量
    gpu_count = torch.cuda.device_count()
    print(f"GPU数量: {gpu_count}")
    
    # 检查环境变量
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
    
    # 列出所有GPU
    print("\nGPU详细信息:")
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} - {props.total_memory//1024**3}GB")
    
    # 测试GPU访问
    print(f"\n测试GPU访问:")
    for i in range(min(gpu_count, 4)):  # 只测试前4个GPU
        try:
            device = torch.device(f'cuda:{i}')
            test_tensor = torch.randn(100, 100, device=device)
            print(f"  GPU {i}: ✓ 可访问")
        except Exception as e:
            print(f"  GPU {i}: ❌ 访问失败 - {e}")
    
    print(f"\n推荐配置:")
    if gpu_count >= 8:
        print(f"  建议使用4-6个GPU进行分布式训练")
        print(f"  命令: python scripts/simple_distributed_train.py --num_gpus 4")
    elif gpu_count >= 4:
        print(f"  建议使用{gpu_count}个GPU进行分布式训练")
        print(f"  命令: python scripts/simple_distributed_train.py --num_gpus {gpu_count}")
    elif gpu_count >= 2:
        print(f"  建议使用{gpu_count}个GPU进行分布式训练")
        print(f"  命令: python scripts/simple_distributed_train.py --num_gpus {gpu_count}")
    else:
        print(f"  建议使用单GPU训练")
        print(f"  命令: python scripts/train.py --gpu_ids 0")

if __name__ == "__main__":
    test_gpu_allocation()