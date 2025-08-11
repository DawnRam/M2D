#!/usr/bin/env python3
"""
PanDerm-Guided Diffusion训练脚本
"""

import os
import sys
import argparse
import torch
import wandb
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.config import Config
from src.training import PanDermDiffusionTrainer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="PanDerm-Guided Diffusion训练")
    
    # 数据相关
    parser.add_argument(
        "--data_root", 
        type=str, 
        default="./data/ISIC",
        help="ISIC数据集根目录"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=16,
        help="批大小"
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=4,
        help="数据加载器工作进程数"
    )
    parser.add_argument(
        "--image_size", 
        type=int, 
        default=224,
        help="图像尺寸"
    )
    
    # 模型相关
    parser.add_argument(
        "--panderm_model", 
        type=str, 
        default="panderm-large",
        choices=["panderm-base", "panderm-large"],
        help="PanDerm模型类型"
    )
    parser.add_argument(
        "--panderm_freeze", 
        action="store_true",
        help="冻结PanDerm参数"
    )
    parser.add_argument(
        "--fusion_type", 
        type=str, 
        default="cross_attention",
        choices=["concat", "add", "cross_attention", "adaptive"],
        help="特征融合方式"
    )
    
    # 训练相关
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=100,
        help="训练轮数"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=1e-4,
        help="学习率"
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=1e-2,
        help="权重衰减"
    )
    parser.add_argument(
        "--mixed_precision", 
        action="store_true",
        help="启用混合精度训练"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", 
        type=int, 
        default=1,
        help="梯度累积步数"
    )
    
    # 损失函数权重
    parser.add_argument(
        "--alpha_diffusion", 
        type=float, 
        default=1.0,
        help="扩散损失权重"
    )
    parser.add_argument(
        "--beta_recon", 
        type=float, 
        default=0.5,
        help="重构损失权重"
    )
    parser.add_argument(
        "--gamma_repa", 
        type=float, 
        default=0.3,
        help="REPA损失权重"
    )
    parser.add_argument(
        "--delta_perceptual", 
        type=float, 
        default=0.2,
        help="感知损失权重"
    )
    
    # Diffusion参数
    parser.add_argument(
        "--num_diffusion_steps", 
        type=int, 
        default=1000,
        help="扩散步数"
    )
    parser.add_argument(
        "--noise_schedule", 
        type=str, 
        default="cosine",
        choices=["linear", "cosine"],
        help="噪声调度"
    )
    
    # 输出和日志
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./outputs",
        help="输出目录"
    )
    parser.add_argument(
        "--checkpoint_dir", 
        type=str, 
        default="./checkpoints",
        help="检查点目录"
    )
    parser.add_argument(
        "--log_dir", 
        type=str, 
        default="./logs",
        help="日志目录"
    )
    parser.add_argument(
        "--experiment_name", 
        type=str, 
        default="panderm_diffusion",
        help="实验名称"
    )
    parser.add_argument(
        "--save_every", 
        type=int, 
        default=10,
        help="保存检查点间隔"
    )
    parser.add_argument(
        "--eval_every", 
        type=int, 
        default=5,
        help="验证间隔"
    )
    parser.add_argument(
        "--log_every", 
        type=int, 
        default=100,
        help="日志记录间隔"
    )
    
    # Wandb
    parser.add_argument(
        "--use_wandb", 
        action="store_true",
        help="启用Wandb日志"
    )
    parser.add_argument(
        "--wandb_project", 
        type=str, 
        default="panderm-diffusion",
        help="Wandb项目名称"
    )
    
    # 恢复训练
    parser.add_argument(
        "--resume_from", 
        type=str, 
        default=None,
        help="从检查点恢复训练"
    )
    
    # 随机种子
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="随机种子"
    )
    
    return parser.parse_args()


def update_config_from_args(config: Config, args) -> Config:
    """根据命令行参数更新配置"""
    
    # 数据配置
    config.data.data_root = args.data_root
    config.data.batch_size = args.batch_size
    config.data.num_workers = args.num_workers
    config.data.image_size = args.image_size
    
    # 模型配置
    config.model.panderm_model = args.panderm_model
    config.model.panderm_freeze = args.panderm_freeze
    config.model.fusion_type = args.fusion_type
    
    # 训练配置
    config.training.epochs = args.epochs
    config.training.learning_rate = args.learning_rate
    config.training.weight_decay = args.weight_decay
    config.training.alpha_diffusion = args.alpha_diffusion
    config.training.beta_reconstruction = args.beta_recon
    config.training.gamma_alignment = args.gamma_repa
    config.training.delta_perceptual = args.delta_perceptual
    config.training.num_diffusion_steps = args.num_diffusion_steps
    config.training.noise_schedule = args.noise_schedule
    config.training.save_every = args.save_every
    config.training.eval_every = args.eval_every
    config.training.log_every = args.log_every
    
    # 设备和加速
    config.mixed_precision = args.mixed_precision
    config.gradient_accumulation_steps = args.gradient_accumulation_steps
    
    # 输出目录
    config.output_dir = args.output_dir
    config.checkpoint_dir = args.checkpoint_dir
    config.log_dir = args.log_dir
    config.experiment_name = args.experiment_name
    
    # 随机种子
    config.seed = args.seed
    
    return config


def main():
    """主函数"""
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 初始化Wandb（如果启用）
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.experiment_name,
            config=vars(args)
        )
    
    # 加载和更新配置
    config = Config()
    config = update_config_from_args(config, args)
    
    print("=" * 50)
    print("PanDerm-Guided Diffusion训练")
    print("=" * 50)
    print(f"实验名称: {config.experiment_name}")
    print(f"数据目录: {config.data.data_root}")
    print(f"输出目录: {config.output_dir}")
    print(f"PanDerm模型: {config.model.panderm_model}")
    print(f"融合方式: {config.model.fusion_type}")
    print(f"训练轮数: {config.training.epochs}")
    print(f"批大小: {config.data.batch_size}")
    print(f"学习率: {config.training.learning_rate}")
    print("=" * 50)
    
    # 检查数据集
    if not os.path.exists(config.data.data_root):
        print(f"错误: 数据集目录不存在 {config.data.data_root}")
        print("请确保ISIC数据集正确放置在指定目录")
        sys.exit(1)
    
    # 初始化训练器
    try:
        trainer = PanDermDiffusionTrainer(config)
        
        # 设置模型、数据、优化器
        print("初始化模型...")
        trainer.setup_models()
        
        print("初始化数据...")
        trainer.setup_data()
        
        print("初始化优化器...")
        trainer.setup_optimizer()
        
        print("设置Accelerator...")
        trainer.setup_accelerator()
        
        # 恢复检查点（如果指定）
        if args.resume_from:
            if os.path.exists(args.resume_from):
                print(f"从检查点恢复: {args.resume_from}")
                trainer.load_checkpoint(args.resume_from)
            else:
                print(f"警告: 检查点文件不存在 {args.resume_from}")
        
        # 开始训练
        print("开始训练...")
        trainer.train()
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        if args.use_wandb:
            wandb.finish()
    
    print("训练完成！")


if __name__ == "__main__":
    main()