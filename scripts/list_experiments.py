#!/usr/bin/env python3
"""
实验管理工具
列出、查看和管理实验
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import ExperimentManager


def list_experiments(experiment_root: str = "/nfs/scratch/eechengyang/Code/logs"):
    """列出所有实验"""
    exp_manager = ExperimentManager(experiment_root=experiment_root, auto_create=False)
    experiments = exp_manager.list_experiments()
    
    if not experiments:
        print("未找到任何实验")
        return
    
    print(f"找到 {len(experiments)} 个实验:")
    print("-" * 80)
    
    for exp in experiments:
        exp_name = exp.get('experiment_name', 'unknown')
        created_time = exp.get('created_time', 'unknown')
        description = exp.get('description', 'PanDerm-Guided Diffusion训练实验')
        
        print(f"实验名称: {exp_name}")
        print(f"创建时间: {created_time}")
        print(f"描述: {description}")
        
        # 检查实验目录是否存在
        exp_dir = Path(experiment_root) / exp_name
        if exp_dir.exists():
            # 统计文件数量
            checkpoint_files = list((exp_dir / "checkpoints").glob("*.pt")) if (exp_dir / "checkpoints").exists() else []
            log_files = list((exp_dir / "logs").glob("*.log")) if (exp_dir / "logs").exists() else []
            
            print(f"状态: ✓ 存在")
            print(f"检查点数量: {len(checkpoint_files)}")
            print(f"日志文件数量: {len(log_files)}")
        else:
            print(f"状态: ❌ 目录不存在")
        
        print("-" * 80)


def show_experiment_details(experiment_name: str, experiment_root: str = "/nfs/scratch/eechengyang/Code/logs"):
    """显示实验详情"""
    exp_dir = Path(experiment_root) / experiment_name
    
    if not exp_dir.exists():
        print(f"❌ 实验目录不存在: {exp_dir}")
        return
    
    print(f"实验详情: {experiment_name}")
    print("=" * 50)
    print(f"实验目录: {exp_dir}")
    
    # 显示目录结构
    subdirs = ["checkpoints", "logs", "outputs", "generated_images", "cache", "wandb", "configs", "code_backup"]
    
    for subdir in subdirs:
        subdir_path = exp_dir / subdir
        if subdir_path.exists():
            if subdir_path.is_dir():
                file_count = len(list(subdir_path.iterdir()))
                print(f"  ✓ {subdir}/: {file_count} 个文件")
            else:
                print(f"  ✓ {subdir}: 文件")
        else:
            print(f"  - {subdir}/: 不存在")
    
    # 显示配置信息
    config_file = exp_dir / "configs" / "config.json"
    if config_file.exists():
        print(f"\n配置文件: {config_file}")
        try:
            import json
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # 显示关键配置
            key_configs = ['epochs', 'batch_size', 'learning_rate', 'experiment_name']
            print("关键配置:")
            for key in key_configs:
                if key in config:
                    print(f"  {key}: {config[key]}")
        except Exception as e:
            print(f"  配置文件读取失败: {e}")
    
    # 显示最新检查点
    checkpoint_dir = exp_dir / "checkpoints"
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
            print(f"\n最新检查点: {latest_checkpoint.name}")
            print(f"文件大小: {latest_checkpoint.stat().st_size / 1024 / 1024:.1f} MB")


def clean_experiments(keep_recent: int = 10, experiment_root: str = "/nfs/scratch/eechengyang/Code/logs"):
    """清理旧实验"""
    exp_manager = ExperimentManager(experiment_root=experiment_root, auto_create=False)
    
    print(f"清理旧实验，保留最近的 {keep_recent} 个...")
    
    try:
        exp_manager.cleanup_old_experiments(keep_recent=keep_recent)
        print("✓ 清理完成")
    except Exception as e:
        print(f"❌ 清理失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="实验管理工具")
    parser.add_argument("command", choices=["list", "show", "clean"], help="操作命令")
    parser.add_argument("experiment_name", nargs="?", help="实验名称（show命令需要）")
    parser.add_argument("--keep", type=int, default=10, help="清理时保留的实验数量")
    parser.add_argument("--experiment_root", type=str, 
                       default="/nfs/scratch/eechengyang/Code/logs", 
                       help="实验根目录")
    
    args = parser.parse_args()
    
    if args.command == "list":
        list_experiments(args.experiment_root)
    
    elif args.command == "show":
        if not args.experiment_name:
            print("❌ show命令需要指定实验名称")
            parser.print_help()
            return
        show_experiment_details(args.experiment_name, args.experiment_root)
    
    elif args.command == "clean":
        clean_experiments(args.keep, args.experiment_root)


if __name__ == "__main__":
    main()