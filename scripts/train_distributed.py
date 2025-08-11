#!/usr/bin/env python3
"""
PanDerm-Guided Diffusion 分布式训练启动脚本
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="分布式训练启动器")
    
    # 数据相关
    parser.add_argument("--data_root", type=str, default="/nfs/scratch/eechengyang/Data/ISIC", 
                       help="数据集根目录")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载器工作进程数")
    
    # 训练相关
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="权重衰减")
    
    # 模型相关
    parser.add_argument("--panderm_freeze", action="store_true", help="冻结PanDerm特征提取器")
    parser.add_argument("--use_ema", action="store_true", help="使用EMA")
    
    # 损失权重
    parser.add_argument("--alpha_diffusion", type=float, default=1.0, help="扩散损失权重")
    parser.add_argument("--beta_recon", type=float, default=0.5, help="重构损失权重")
    parser.add_argument("--gamma_repa", type=float, default=0.3, help="REPA对齐损失权重")
    parser.add_argument("--delta_perceptual", type=float, default=0.2, help="感知损失权重")
    
    # 训练设置
    parser.add_argument("--mixed_precision", action="store_true", help="启用混合精度训练")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数")
    
    # GPU和分布式
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3", help="GPU设备ID")
    parser.add_argument("--num_processes", type=int, default=-1, help="进程数")
    
    # 实验管理
    parser.add_argument("--experiment_name", type=str, default="panderm_diffusion", help="实验名称")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    # WandB
    parser.add_argument("--use_wandb", action="store_true", help="启用WandB记录")
    parser.add_argument("--wandb_project", type=str, default="panderm-diffusion", help="WandB项目名称")
    
    return parser.parse_args()


def get_gpu_count():
    """获取可用的GPU数量"""
    try:
        import torch
        return torch.cuda.device_count()
    except:
        return 0


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 50)
    print("PanDerm-Guided Diffusion 分布式训练")
    print("=" * 50)
    
    # 检测GPU数量
    gpu_count = get_gpu_count()
    if gpu_count < 2:
        print(f"⚠ 警告: 检测到{gpu_count}个GPU，建议使用单GPU训练模式")
        print("使用命令: python scripts/train.py")
        return
    
    # 确定进程数
    if args.num_processes == -1:
        num_processes = gpu_count
    else:
        num_processes = min(args.num_processes, gpu_count)
    
    print(f"✓ 使用{num_processes}个GPU进行分布式训练")
    
    # 构建accelerate launch命令
    cmd = [
        "accelerate", "launch",
        "--config_file", "default_config.yaml",
        "--num_processes", str(num_processes),
        "--multi_gpu",
        "--mixed_precision", "fp16" if args.mixed_precision else "no",
        "scripts/train.py"
    ]
    
    # 添加训练参数
    train_args = [
        "--data_root", args.data_root,
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--epochs", str(args.epochs),
        "--learning_rate", str(args.learning_rate),
        "--weight_decay", str(args.weight_decay),
        "--alpha_diffusion", str(args.alpha_diffusion),
        "--beta_recon", str(args.beta_recon),
        "--gamma_repa", str(args.gamma_repa),
        "--delta_perceptual", str(args.delta_perceptual),
        "--experiment_name", args.experiment_name,
        "--seed", str(args.seed),
        "--gpu_ids", args.gpu_ids,
        "--distributed",
        "--num_processes", str(num_processes)
    ]
    
    # 添加可选参数
    if args.panderm_freeze:
        train_args.append("--panderm_freeze")
    if args.use_ema:
        train_args.append("--use_ema")
    if args.mixed_precision:
        train_args.append("--mixed_precision")
    if args.use_wandb:
        train_args.extend(["--use_wandb", "--wandb_project", args.wandb_project])
    
    if args.gradient_accumulation_steps > 1:
        train_args.extend(["--gradient_accumulation_steps", str(args.gradient_accumulation_steps)])
    
    # 组合完整命令
    full_cmd = cmd + train_args
    
    print("执行命令:")
    print(" ".join(full_cmd))
    print()
    
    # 执行训练
    try:
        result = subprocess.run(full_cmd, check=True)
        print("✓ 分布式训练完成！")
    except subprocess.CalledProcessError as e:
        print(f"❌ 训练失败，错误码: {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ 未找到accelerate命令，请先安装: pip install accelerate")
        print("然后运行: accelerate config")
        sys.exit(1)


if __name__ == "__main__":
    main()