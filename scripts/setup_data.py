#!/usr/bin/env python3
"""
数据和环境设置脚本
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.model_paths import DATA_ROOT, EXPERIMENT_ROOT


def verify_setup():
    """验证设置"""
    
    print("\n" + "="*50)
    print("验证设置...")
    
    # 检查数据目录结构 - 7个类别子目录
    expected_categories = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']
    total_images = 0
    
    for category in expected_categories:
        category_dir = os.path.join(DATA_ROOT, category)
        if os.path.exists(category_dir):
            image_files = [f for f in os.listdir(category_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            total_images += len(image_files)
            print(f"✓ {category} 目录: {len(image_files)} 张图像")
        else:
            print(f"✗ {category} 目录不存在")
    
    print(f"✓ 总图像数量: {total_images}")
    if total_images > 0:
        print("✓ 发现ISIC数据集图像")
        # 显示类别分布
        category_counts = {}
        for category in expected_categories:
            category_dir = os.path.join(DATA_ROOT, category)
            if os.path.exists(category_dir):
                count = len([f for f in os.listdir(category_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                category_counts[category] = count
        print(f"✓ 实际类别分布: {category_counts}")
    else:
        print("⚠ 未发现图像数据")
        
    # 检查旧的images目录（如果存在，提醒用户）
    old_images_dir = os.path.join(DATA_ROOT, "images")
    if os.path.exists(old_images_dir):
        print(f"⚠ 发现旧的images目录，建议删除: {old_images_dir}")
    
    # 不再检查metadata.csv文件，直接从目录结构获取类别分布
    print("✓ 使用真实ISIC数据集，无需元数据文件")
    
    # 检查实验根目录
    print(f"\n检查实验根目录: {EXPERIMENT_ROOT}")
    if os.path.exists(EXPERIMENT_ROOT):
        print(f"✓ 实验根目录存在: {EXPERIMENT_ROOT}")
    else:
        print(f"⚠ 实验根目录不存在，将自动创建: {EXPERIMENT_ROOT}")
        os.makedirs(EXPERIMENT_ROOT, exist_ok=True)
        print(f"✓ 已创建实验根目录: {EXPERIMENT_ROOT}")
    
    print("="*50)
    return True


def download_pretrained_weights():
    """下载预训练权重（占位函数）"""
    print("配置预训练模型...")
    
    # 创建模型目录
    models_dir = os.path.join(project_root, "models")
    os.makedirs(os.path.join(models_dir, "panderm"), exist_ok=True)
    os.makedirs(os.path.join(models_dir, "pretrained"), exist_ok=True)
    
    print("PanDerm模型路径已在配置中设置")
    print("系统会在训练时自动下载必要的预训练权重")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="数据和环境设置")
    parser.add_argument("--verify", action="store_true", help="验证设置")
    parser.add_argument("--download_models", action="store_true", help="下载预训练模型")
    parser.add_argument("--all", action="store_true", help="执行所有设置步骤")
    
    args = parser.parse_args()
    
    if args.all:
        args.download_models = True
        args.verify = True
    
    if not any([args.download_models, args.verify]):
        args.all = True
        args.download_models = True
        args.verify = True
    
    print("PanDerm-Guided Diffusion 数据设置")
    print("="*50)
    
    try:
        if args.download_models:
            download_pretrained_weights()
        
        if args.verify:
            success = verify_setup()
            if success:
                print("\n🎉 设置完成！现在可以开始训练了")
                print("\n快速开始:")
                print("1. bash run_training.sh       # 一键训练")
                print("2. python scripts/train.py    # 手动训练")
                print("3. python scripts/list_experiments.py list  # 查看实验")
            else:
                print("\n❌ 设置验证失败，请检查数据路径")
                sys.exit(1)
                
    except Exception as e:
        print(f"\n❌ 设置过程中出现错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()