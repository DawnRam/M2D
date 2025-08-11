#!/usr/bin/env python3
"""
实验管理脚本
列出、查看和管理实验结果
"""

import sys
import os
from pathlib import Path
import argparse
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import ExperimentManager


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="实验管理工具")
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 列出实验
    list_parser = subparsers.add_parser('list', help='列出所有实验')
    list_parser.add_argument('--limit', type=int, default=10, help='显示数量限制')
    
    # 查看实验详情
    show_parser = subparsers.add_parser('show', help='查看实验详情')
    show_parser.add_argument('experiment_name', help='实验名称')
    
    # 清理旧实验
    clean_parser = subparsers.add_parser('clean', help='清理旧实验')
    clean_parser.add_argument('--keep', type=int, default=5, help='保留最近的实验数量')
    clean_parser.add_argument('--force', action='store_true', help='强制删除，不询问确认')
    
    # 比较实验
    compare_parser = subparsers.add_parser('compare', help='比较多个实验')
    compare_parser.add_argument('experiments', nargs='+', help='要比较的实验名称')
    
    return parser.parse_args()


def list_experiments(limit: int = 10):
    """列出实验"""
    exp_manager = ExperimentManager(auto_create=False)
    experiments = exp_manager.list_experiments()
    
    if not experiments:
        print("未找到任何实验")
        return
    
    print(f"\n找到 {len(experiments)} 个实验（显示最近 {min(limit, len(experiments))} 个）:")
    print("=" * 80)
    print(f"{'实验名称':<30} {'创建时间':<20} {'状态':<10} {'备注':<20}")
    print("=" * 80)
    
    for i, exp in enumerate(experiments[:limit]):
        name = exp['experiment_name']
        created_time = exp.get('created_time', 'unknown')
        if created_time != 'unknown':
            try:
                dt = datetime.fromisoformat(created_time.replace('Z', '+00:00'))
                created_time = dt.strftime('%Y-%m-%d %H:%M')
            except:
                pass
        
        # 检查实验状态
        exp_dir = Path(exp['experiment_dir'])
        if exp_dir.exists():
            checkpoint_dir = exp_dir / "checkpoints"
            if (checkpoint_dir / "best_model.pt").exists():
                status = "完成"
            elif any(checkpoint_dir.glob("checkpoint_*.pt")):
                status = "进行中"
            else:
                status = "开始"
        else:
            status = "丢失"
        
        description = exp.get('description', '')[:18]
        
        print(f"{name:<30} {created_time:<20} {status:<10} {description:<20}")
    
    if len(experiments) > limit:
        print(f"\n... 还有 {len(experiments) - limit} 个实验（使用 --limit 查看更多）")


def show_experiment(experiment_name: str):
    """显示实验详情"""
    exp_manager = ExperimentManager(auto_create=False)
    experiments = exp_manager.list_experiments()
    
    # 查找实验
    target_exp = None
    for exp in experiments:
        if exp['experiment_name'] == experiment_name:
            target_exp = exp
            break
    
    if not target_exp:
        print(f"未找到实验: {experiment_name}")
        return
    
    # 显示基本信息
    print(f"\n实验详情: {experiment_name}")
    print("=" * 60)
    
    exp_dir = Path(target_exp['experiment_dir'])
    
    print(f"实验目录: {exp_dir}")
    print(f"创建时间: {target_exp.get('created_time', 'unknown')}")
    print(f"描述: {target_exp.get('description', '无')}")
    print(f"Git提交: {target_exp.get('git_commit', 'unknown')[:8]}...")
    
    # 检查文件
    print(f"\n文件状态:")
    
    files_to_check = {
        "配置文件": exp_dir / "configs" / "config.json",
        "最佳模型": exp_dir / "checkpoints" / "best_model.pt",
        "训练日志": exp_dir / "logs" / "training.log",
        "实验信息": exp_dir / "experiment_info.json"
    }
    
    for name, path in files_to_check.items():
        if path.exists():
            if path.is_file():
                size = path.stat().st_size / 1024 / 1024  # MB
                print(f"  ✓ {name}: {size:.1f}MB")
            else:
                print(f"  ✓ {name}: 存在")
        else:
            print(f"  ✗ {name}: 缺失")
    
    # 检查检查点
    checkpoint_dir = exp_dir / "checkpoints"
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
        print(f"  ✓ 检查点文件: {len(checkpoints)} 个")
    
    # 检查生成图像
    generated_dir = exp_dir / "generated_images"
    if generated_dir.exists():
        images = list(generated_dir.glob("*.png")) + list(generated_dir.glob("*.jpg"))
        print(f"  ✓ 生成图像: {len(images)} 张")
    
    # 显示配置（如果存在）
    config_file = exp_dir / "configs" / "config.json"
    if config_file.exists():
        try:
            import json
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            print(f"\n主要配置:")
            important_keys = [
                'training.epochs',
                'training.learning_rate', 
                'data.batch_size',
                'model.panderm_model',
                'experiment_name'
            ]
            
            for key in important_keys:
                value = config
                for k in key.split('.'):
                    if isinstance(value, dict) and k in value:
                        value = value[k]
                    else:
                        value = "未设置"
                        break
                print(f"  {key}: {value}")
                
        except Exception as e:
            print(f"  配置文件读取失败: {e}")


def clean_experiments(keep: int = 5, force: bool = False):
    """清理旧实验"""
    exp_manager = ExperimentManager(auto_create=False)
    experiments = exp_manager.list_experiments()
    
    if len(experiments) <= keep:
        print(f"实验数量({len(experiments)})未超过保留数量({keep})，无需清理")
        return
    
    to_remove = experiments[keep:]
    
    print(f"\n将删除 {len(to_remove)} 个旧实验:")
    for exp in to_remove:
        print(f"  - {exp['experiment_name']} ({exp.get('created_time', 'unknown')})")
    
    if not force:
        confirm = input(f"\n确认删除这些实验？(y/N): ")
        if confirm.lower() != 'y':
            print("取消删除")
            return
    
    # 执行删除
    deleted_count = 0
    for exp in to_remove:
        exp_dir = Path(exp['experiment_dir'])
        if exp_dir.exists():
            try:
                import shutil
                shutil.rmtree(exp_dir)
                print(f"✓ 已删除: {exp['experiment_name']}")
                deleted_count += 1
            except Exception as e:
                print(f"✗ 删除失败: {exp['experiment_name']}, 错误: {e}")
    
    print(f"\n清理完成，删除了 {deleted_count} 个实验")


def compare_experiments(experiment_names: list):
    """比较实验"""
    exp_manager = ExperimentManager(auto_create=False)
    experiments = exp_manager.list_experiments()
    
    # 查找实验
    found_experiments = []
    for name in experiment_names:
        for exp in experiments:
            if exp['experiment_name'] == name:
                found_experiments.append(exp)
                break
        else:
            print(f"未找到实验: {name}")
            return
    
    print(f"\n比较 {len(found_experiments)} 个实验:")
    print("=" * 100)
    
    # 表头
    headers = ["实验名称", "创建时间", "状态", "配置差异"]
    col_widths = [25, 18, 10, 40]
    
    header_line = ""
    for header, width in zip(headers, col_widths):
        header_line += f"{header:<{width}} "
    print(header_line)
    print("=" * 100)
    
    # 比较每个实验
    for exp in found_experiments:
        name = exp['experiment_name'][:24]
        
        created_time = exp.get('created_time', 'unknown')
        if created_time != 'unknown':
            try:
                dt = datetime.fromisoformat(created_time.replace('Z', '+00:00'))
                created_time = dt.strftime('%m-%d %H:%M')
            except:
                pass
        
        # 检查状态
        exp_dir = Path(exp['experiment_dir'])
        if (exp_dir / "checkpoints" / "best_model.pt").exists():
            status = "完成"
        elif any((exp_dir / "checkpoints").glob("checkpoint_*.pt")):
            status = "进行中"
        else:
            status = "开始"
        
        # 获取关键配置
        config_summary = "配置缺失"
        config_file = exp_dir / "configs" / "config.json"
        if config_file.exists():
            try:
                import json
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # 提取关键配置
                epochs = config.get('training', {}).get('epochs', '?')
                lr = config.get('training', {}).get('learning_rate', '?')
                batch_size = config.get('data', {}).get('batch_size', '?')
                
                config_summary = f"ep:{epochs} lr:{lr} bs:{batch_size}"
            except:
                config_summary = "配置读取失败"
        
        # 输出行
        line = f"{name:<25} {created_time:<18} {status:<10} {config_summary:<40}"
        print(line)


def main():
    """主函数"""
    args = parse_args()
    
    if args.command is None:
        print("请指定命令，使用 --help 查看帮助")
        return
    
    try:
        if args.command == 'list':
            list_experiments(args.limit)
        
        elif args.command == 'show':
            show_experiment(args.experiment_name)
        
        elif args.command == 'clean':
            clean_experiments(args.keep, args.force)
        
        elif args.command == 'compare':
            compare_experiments(args.experiments)
        
    except KeyboardInterrupt:
        print("\n操作被用户中断")
    except Exception as e:
        print(f"操作失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()