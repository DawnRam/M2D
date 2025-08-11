#!/usr/bin/env python3
"""
PanDerm-Guided Diffusion 训练脚本
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.config import Config
from src.training.trainer import PanDermDiffusionTrainer
from src.utils import create_experiment_manager


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="PanDerm-Guided Diffusion训练")
    
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
    parser.add_argument("--device", type=str, default="auto", help="设备")
    parser.add_argument("--gpu_ids", type=str, default="0", help="GPU设备ID")
    parser.add_argument("--distributed", action="store_true", help="启用分布式训练")
    parser.add_argument("--num_processes", type=int, default=-1, help="进程数")
    
    # 实验管理
    parser.add_argument("--experiment_name", type=str, default="panderm_diffusion", help="实验名称")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    # WandB
    parser.add_argument("--use_wandb", action="store_true", help="启用WandB记录")
    parser.add_argument("--wandb_project", type=str, default="panderm-diffusion", help="WandB项目名称")
    
    # 输出目录（这些会被实验管理器覆盖）
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录（自动设置）")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="检查点目录（自动设置）")
    parser.add_argument("--log_dir", type=str, default=None, help="日志目录（自动设置）")
    
    return parser.parse_args()


def update_config_from_args(config: Config, args) -> Config:
    """根据命令行参数更新配置"""
    
    # 数据配置
    config.data.data_root = args.data_root
    config.data.batch_size = args.batch_size
    config.data.num_workers = args.num_workers
    
    # 训练配置
    config.training.epochs = args.epochs
    config.training.learning_rate = args.learning_rate
    config.training.weight_decay = args.weight_decay
    config.training.use_ema = args.use_ema
    
    # 损失权重
    config.training.alpha_diffusion = args.alpha_diffusion
    config.training.beta_recon = args.beta_recon
    config.training.gamma_repa = args.gamma_repa
    config.training.delta_perceptual = args.delta_perceptual
    
    # 模型配置
    config.model.freeze_panderm = args.panderm_freeze
    
    # 设备配置
    config.device = args.device
    config.gpu_ids = args.gpu_ids
    config.distributed = args.distributed
    config.num_processes = args.num_processes
    config.mixed_precision = args.mixed_precision
    config.gradient_accumulation_steps = args.gradient_accumulation_steps
    
    # 输出目录（如果参数中指定了路径，则使用；否则保持默认，由实验管理器覆盖）
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir
    if args.log_dir is not None:
        config.log_dir = args.log_dir
    config.experiment_name = args.experiment_name
    
    # 随机种子
    config.seed = args.seed
    
    return config


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 加载基础配置
    config = Config()
    
    # 根据参数更新配置
    config = update_config_from_args(config, args)
    
    print("=" * 50)
    print("PanDerm-Guided Diffusion 训练开始")
    print("=" * 50)
    
    # 创建实验管理器
    print("创建实验管理器...")
    exp_manager = create_experiment_manager(experiment_name=args.experiment_name)
    
    print(f"✓ 实验目录: {exp_manager.experiment_dir}")
    
    # 更新配置中的目录路径为实验目录
    config.output_dir = str(exp_manager.get_output_dir())
    config.checkpoint_dir = str(exp_manager.get_checkpoint_dir())
    config.log_dir = str(exp_manager.get_log_dir())
    
    # 保存配置到实验目录
    exp_manager.save_config(config.__dict__)
    
    # 初始化WandB（如果启用）
    if args.use_wandb:
        # 设置WandB目录到实验目录
        wandb_dir = exp_manager.get_dir("wandb")
        os.environ["WANDB_DIR"] = str(wandb_dir)
        
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=exp_manager.experiment_name,
            config=config.__dict__,
            dir=str(wandb_dir)
        )
        print("✓ WandB初始化完成")
    
    # 初始化训练器
    print("初始化训练器...")
    trainer = PanDermDiffusionTrainer(config)
    
    # 设置模型、数据、优化器
    trainer.setup_models()
    trainer.setup_data()
    trainer.setup_optimizer()
    trainer.setup_accelerator()
    
    print("✓ 训练器设置完成")
    print(f"✓ 实验结果将保存到: {exp_manager.experiment_dir}")
    
    # 开始训练
    print("\n开始训练...")
    trainer.train()
    
    print("✓ 训练完成！")
    print(f"✓ 实验结果保存在: {exp_manager.experiment_dir}")


if __name__ == "__main__":
    main()