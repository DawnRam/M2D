#!/usr/bin/env python3
"""
修正版分布式训练脚本 - 解决GPU重复分配问题
参考REPA-E的最佳实践
"""

import os
import sys
import subprocess
import argparse
import torch
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    parser = argparse.ArgumentParser(description="修正版分布式训练")
    parser.add_argument("--num_gpus", type=int, default=2, help="使用的GPU数量")
    parser.add_argument("--gpu_ids", type=str, default="0,1", help="指定GPU ID，如'0,1,2,3'")
    parser.add_argument("--data_root", type=str, default="/nfs/scratch/eechengyang/Data/ISIC", help="数据根目录")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--experiment_name", type=str, default="fixed-distributed", help="实验名称")
    args = parser.parse_args()
    
    print("=" * 60)
    print("修正版分布式训练启动")
    print("=" * 60)
    
    # 解析GPU ID
    if args.gpu_ids:
        gpu_list = [int(x.strip()) for x in args.gpu_ids.split(',')]
        args.num_gpus = len(gpu_list)
    else:
        gpu_list = list(range(args.num_gpus))
    
    print(f"✓ 使用GPU: {gpu_list}")
    print(f"✓ GPU数量: {args.num_gpus}")
    
    # 验证GPU可用性
    available_gpus = torch.cuda.device_count()
    print(f"✓ 系统可用GPU总数: {available_gpus}")
    
    for gpu_id in gpu_list:
        if gpu_id >= available_gpus:
            print(f"❌ GPU {gpu_id} 不存在，系统只有 {available_gpus} 个GPU")
            return 1
    
    # 设置环境变量，明确指定使用的GPU
    gpu_ids_str = ','.join(map(str, gpu_list))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids_str
    
    # 设置NCCL调试信息
    env["NCCL_DEBUG"] = "INFO"
    env["NCCL_SOCKET_IFNAME"] = "lo"  # 使用本地回环接口
    env["NCCL_P2P_DISABLE"] = "1"    # 禁用P2P，避免某些硬件问题
    
    print(f"✓ 设置 CUDA_VISIBLE_DEVICES={gpu_ids_str}")
    
    # 首先生成合适的accelerate配置
    print("生成accelerate配置文件...")
    try:
        setup_result = subprocess.run([
            sys.executable, "scripts/setup_accelerate_config.py",
            "--num_gpus", str(args.num_gpus),
            "--mixed_precision", "fp16",
            "--output", "distributed_config.yaml"
        ], check=True, capture_output=True, text=True, cwd=project_root)
        print("✓ accelerate配置文件已生成")
    except subprocess.CalledProcessError as e:
        print(f"⚠ 配置文件生成失败: {e}")
        print("使用accelerate launch方式...")
    
    # 使用accelerate launch（推荐方式，参考REPA-E）
    cmd = [
        "accelerate", "launch",
        "--config_file", "distributed_config.yaml",
        "scripts/train.py",
        "--data_root", args.data_root,
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--experiment_name", args.experiment_name,
        "--distributed",
        "--num_processes", str(args.num_gpus),
        "--panderm_freeze"
    ]
    
    print("\n执行命令:")
    print(" ".join(cmd))
    print(f"\n环境变量:")
    print(f"  CUDA_VISIBLE_DEVICES={gpu_ids_str}")
    print(f"  NCCL_DEBUG=INFO")
    print(f"  NCCL_P2P_DISABLE=1")
    print()
    
    try:
        # 执行训练
        result = subprocess.run(cmd, env=env, check=True)
        print("✅ 分布式训练完成")
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 分布式训练失败，退出码: {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n⚠ 训练被用户中断")
        return 1

if __name__ == "__main__":
    sys.exit(main())