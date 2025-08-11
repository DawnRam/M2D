#!/usr/bin/env python3
"""
动态生成accelerate配置文件
"""

import os
import sys
import yaml
import torch
from pathlib import Path

def detect_gpu_count():
    """检测可用GPU数量"""
    try:
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        else:
            return 0
    except Exception:
        return 0

def generate_accelerate_config(num_gpus, mixed_precision="fp16", output_path="default_config.yaml"):
    """生成accelerate配置"""
    
    if num_gpus <= 0:
        # CPU配置
        config = {
            "compute_environment": "LOCAL_MACHINE",
            "distributed_type": "NO",
            "downcast_bf16": "no",
            "machine_rank": 0,
            "main_training_function": "main",
            "mixed_precision": "no",
            "num_machines": 1,
            "num_processes": 1,
            "use_cpu": True
        }
    elif num_gpus == 1:
        # 单GPU配置
        config = {
            "compute_environment": "LOCAL_MACHINE", 
            "distributed_type": "NO",
            "downcast_bf16": "no",
            "gpu_ids": "0",
            "machine_rank": 0,
            "main_training_function": "main",
            "mixed_precision": mixed_precision,
            "num_machines": 1,
            "num_processes": 1,
            "use_cpu": False
        }
    else:
        # 多GPU配置
        config = {
            "compute_environment": "LOCAL_MACHINE",
            "distributed_type": "MULTI_GPU", 
            "downcast_bf16": "no",
            "gpu_ids": "all",
            "machine_rank": 0,
            "main_training_function": "main",
            "mixed_precision": mixed_precision,
            "num_machines": 1,
            "num_processes": min(num_gpus, 8),  # 最多使用8个GPU
            "rdzv_backend": "static",
            "same_network": True,
            "tpu_env": [],
            "tpu_use_cluster": False,
            "tpu_use_sudo": False,
            "use_cpu": False
        }
    
    # 保存配置文件
    project_root = Path(__file__).parent.parent
    config_path = project_root / output_path
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    return config_path, config

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="生成accelerate配置文件")
    parser.add_argument("--num_gpus", type=int, default=-1, help="GPU数量 (-1表示自动检测)")
    parser.add_argument("--mixed_precision", type=str, default="fp16", 
                       choices=["no", "fp16", "bf16"], help="混合精度类型")
    parser.add_argument("--output", type=str, default="default_config.yaml", 
                       help="输出配置文件名")
    
    args = parser.parse_args()
    
    # 检测或使用指定的GPU数量
    if args.num_gpus == -1:
        num_gpus = detect_gpu_count()
        print(f"🔍 自动检测GPU数量: {num_gpus}")
    else:
        num_gpus = args.num_gpus
        print(f"📝 使用指定GPU数量: {num_gpus}")
    
    # 生成配置
    config_path, config = generate_accelerate_config(
        num_gpus, args.mixed_precision, args.output
    )
    
    print(f"✅ 配置文件已生成: {config_path}")
    print(f"📊 配置内容:")
    print(f"   - 分布式类型: {config.get('distributed_type', 'N/A')}")
    print(f"   - 进程数: {config.get('num_processes', 'N/A')}")
    print(f"   - 混合精度: {config.get('mixed_precision', 'N/A')}")
    print(f"   - GPU设备: {config.get('gpu_ids', 'N/A')}")
    
    # 建议使用命令
    if num_gpus > 1:
        print(f"\n🚀 推荐分布式训练命令:")
        print(f"   accelerate launch --config_file {args.output} scripts/train.py [训练参数]")
    elif num_gpus == 1:
        print(f"\n🚀 推荐单GPU训练命令:")
        print(f"   python scripts/train.py [训练参数]")
    else:
        print(f"\n🚀 推荐CPU训练命令:")
        print(f"   python scripts/train.py --device cpu [训练参数]")

if __name__ == "__main__":
    main()