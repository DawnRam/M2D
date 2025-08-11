#!/usr/bin/env python3
"""
简化的分布式训练脚本 - 避免GPU分配冲突
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="简化分布式训练")
    parser.add_argument("--num_gpus", type=int, default=4, help="使用的GPU数量")
    parser.add_argument("--data_root", type=str, default="/nfs/scratch/eechengyang/Data/ISIC", help="数据根目录")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--experiment_name", type=str, default="distributed-test", help="实验名称")
    args = parser.parse_args()
    
    print("=" * 50)
    print("简化分布式训练启动")
    print("=" * 50)
    
    # 检查GPU数量
    try:
        import torch
        available_gpus = torch.cuda.device_count()
        print(f"✓ 系统可用GPU数量: {available_gpus}")
        
        if args.num_gpus > available_gpus:
            print(f"⚠ 请求GPU数量({args.num_gpus})超过可用数量({available_gpus})")
            args.num_gpus = available_gpus
            
        print(f"✓ 使用GPU数量: {args.num_gpus}")
        
    except ImportError:
        print("❌ PyTorch未安装")
        return 1
    
    # 构建简化的torchrun命令
    cmd = [
        "torchrun",
        "--nproc_per_node", str(args.num_gpus),
        "--nnodes", "1",
        "--node_rank", "0",
        "--master_addr", "localhost", 
        "--master_port", "12355",
        "scripts/train.py",
        "--data_root", args.data_root,
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--experiment_name", args.experiment_name,
        "--distributed",
        "--num_processes", str(args.num_gpus),
        "--panderm_freeze"
    ]
    
    print("执行命令:")
    print(" ".join(cmd))
    print()
    
    # 设置环境变量
    env = os.environ.copy()
    
    # 设置PyTorch分布式环境变量（调试模式）
    env["NCCL_DEBUG"] = "WARN"  # 减少日志噪音
    env["TORCH_DISTRIBUTED_DEBUG"] = "OFF"  # 关闭详细调试
    env["PYTHONPATH"] = str(Path(__file__).parent.parent)  # 添加项目根目录到Python路径
    
    # 让accelerate自动管理CUDA设备
    print(f"CUDA_VISIBLE_DEVICES: {env.get('CUDA_VISIBLE_DEVICES', '未设置')}")
    
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