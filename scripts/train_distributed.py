#!/usr/bin/env python3
"""
分布式训练启动脚本
支持多GPU训练的便捷启动器
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="PanDerm-Guided Diffusion分布式训练")
    
    # GPU相关
    parser.add_argument(
        "--gpu_ids", 
        type=str, 
        default="0,1,2,3",
        help="GPU设备ID，逗号分隔，如 '0,1,2,3' 或 '0'"
    )
    parser.add_argument(
        "--num_processes", 
        type=int, 
        default=-1,
        help="进程数，-1表示自动检测GPU数量"
    )
    
    # 训练参数（传递给train.py）
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=50,
        help="训练轮数"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=16,
        help="批大小"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=1e-4,
        help="学习率"
    )
    parser.add_argument(
        "--panderm_freeze", 
        action="store_true",
        help="冻结PanDerm参数"
    )
    parser.add_argument(
        "--mixed_precision", 
        action="store_true",
        default=True,
        help="启用混合精度训练"
    )
    parser.add_argument(
        "--use_wandb", 
        action="store_true",
        help="启用WandB记录"
    )
    parser.add_argument(
        "--experiment_name", 
        type=str, 
        default="distributed-training",
        help="实验名称"
    )
    
    return parser.parse_args()


def check_gpu_availability(gpu_ids_str: str):
    """检查GPU可用性"""
    import torch
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，无法进行GPU训练")
        return False, 0
    
    total_gpus = torch.cuda.device_count()
    print(f"✓ 系统总共有 {total_gpus} 个GPU")
    
    if gpu_ids_str == "auto":
        gpu_ids = list(range(total_gpus))
    else:
        try:
            gpu_ids = [int(x.strip()) for x in gpu_ids_str.split(',')]
        except ValueError:
            print(f"❌ 无效的GPU ID格式: {gpu_ids_str}")
            return False, 0
    
    # 检查GPU ID是否有效
    invalid_ids = [gid for gid in gpu_ids if gid >= total_gpus]
    if invalid_ids:
        print(f"❌ 无效的GPU ID: {invalid_ids}，系统只有 {total_gpus} 个GPU")
        return False, 0
    
    # 检查GPU内存
    for gpu_id in gpu_ids:
        gpu_name = torch.cuda.get_device_name(gpu_id)
        gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
        print(f"  GPU {gpu_id}: {gpu_name} ({gpu_memory:.1f}GB)")
        
        if gpu_memory < 6.0:  # 最少6GB内存
            print(f"⚠ GPU {gpu_id} 内存不足({gpu_memory:.1f}GB)，建议至少8GB")
    
    return True, len(gpu_ids)


def run_distributed_training(args):
    """运行分布式训练"""
    
    # 检查GPU
    gpu_available, num_gpus = check_gpu_availability(args.gpu_ids)
    if not gpu_available:
        return False
    
    # 设置进程数
    if args.num_processes == -1:
        args.num_processes = num_gpus
    
    print(f"\n{'='*60}")
    print(f"启动分布式训练")
    print(f"{'='*60}")
    print(f"GPU设备: {args.gpu_ids}")
    print(f"进程数: {args.num_processes}")
    print(f"批大小: {args.batch_size}")
    print(f"训练轮数: {args.epochs}")
    print(f"混合精度: {args.mixed_precision}")
    print(f"{'='*60}\n")
    
    # 构建accelerate launch命令
    cmd = [
        "accelerate", "launch",
        "--num_processes", str(args.num_processes),
        "--multi_gpu",
        "--mixed_precision", "fp16" if args.mixed_precision else "no",
    ]
    
    # 添加训练脚本和参数
    train_script = str(project_root / "scripts" / "train.py")
    cmd.extend([
        train_script,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
        "--gpu_ids", args.gpu_ids,
        "--num_processes", str(args.num_processes),
        "--distributed",
        "--experiment_name", f"{args.experiment_name}_distributed",
    ])
    
    # 可选参数
    if args.panderm_freeze:
        cmd.append("--panderm_freeze")
    if args.mixed_precision:
        cmd.append("--mixed_precision")
    if args.use_wandb:
        cmd.append("--use_wandb")
    
    # 设置环境变量
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    env["PYTHONPATH"] = f"{project_root}:{env.get('PYTHONPATH', '')}"
    
    print("执行命令:")
    print(" ".join(cmd))
    print()
    
    try:
        # 运行训练
        result = subprocess.run(cmd, env=env, check=True)
        print("\n🎉 分布式训练完成！")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 训练失败，退出码: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n⚠ 训练被用户中断")
        return False
    except Exception as e:
        print(f"\n❌ 训练异常: {e}")
        return False


def check_accelerate():
    """检查accelerate是否安装"""
    try:
        import accelerate
        print(f"✓ Accelerate版本: {accelerate.__version__}")
        
        # 检查accelerate配置
        result = subprocess.run(
            ["accelerate", "env"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        
        if result.returncode == 0:
            print("✓ Accelerate环境配置正常")
            return True
        else:
            print("⚠ Accelerate环境配置异常，请运行: accelerate config")
            return False
            
    except ImportError:
        print("❌ 未安装accelerate，请运行: pip install accelerate")
        return False
    except subprocess.TimeoutExpired:
        print("⚠ Accelerate环境检查超时")
        return True
    except Exception as e:
        print(f"⚠ Accelerate检查异常: {e}")
        return True


def main():
    """主函数"""
    print("\n" + "="*60)
    print("PanDerm-Guided Diffusion 分布式训练启动器")
    print("="*60)
    
    # 解析参数
    args = parse_args()
    
    # 检查环境
    print("\n检查训练环境...")
    
    # 检查accelerate
    if not check_accelerate():
        print("\n请先配置accelerate环境:")
        print("  pip install accelerate")
        print("  accelerate config")
        return False
    
    # 检查conda环境
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if conda_env:
        print(f"✓ 当前conda环境: {conda_env}")
    else:
        print("⚠ 未检测到conda环境")
    
    # 运行分布式训练
    success = run_distributed_training(args)
    
    if success:
        print("\n" + "="*60)
        print("训练完成！检查结果:")
        print("- 检查点: ./checkpoints/")
        print("- 日志: ./logs/")  
        print("- 生成图像: ./outputs/")
        if args.use_wandb:
            print("- WandB面板: https://wandb.ai")
        print("="*60)
    
    return success


if __name__ == "__main__":
    sys.exit(0 if main() else 1)