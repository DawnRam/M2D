#!/usr/bin/env python3
"""
PanDerm-Guided Diffusion主程序
医学图像数据集扩充项目
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="PanDerm-Guided Diffusion: 医学图像数据集扩充",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

  训练模型:
    python main.py train --data_root ./data/ISIC --epochs 100 --batch_size 16

  生成图像:
    python main.py generate --checkpoint ./checkpoints/best_model.pt --num_samples 10

  图像到图像:
    python main.py image2image --checkpoint ./checkpoints/best_model.pt \\
                               --init_images ./test_images --strength 0.75

  评估质量:
    python main.py evaluate --real_images ./real_images \\
                           --generated_images ./generated_images

更多详细参数请查看各个脚本的帮助信息:
  python scripts/train.py --help
  python scripts/generate.py --help
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--config', type=str, help='配置文件路径（可选）')
    
    # 生成命令
    generate_parser = subparsers.add_parser('generate', help='生成图像')
    generate_parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    generate_parser.add_argument('--num_samples', type=int, default=8, help='生成样本数量')
    
    # 图像到图像命令
    img2img_parser = subparsers.add_parser('image2image', help='图像到图像生成')
    img2img_parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    img2img_parser.add_argument('--init_images', type=str, required=True, help='初始图像路径')
    img2img_parser.add_argument('--strength', type=float, default=0.75, help='变形强度')
    
    # 插值命令
    interp_parser = subparsers.add_parser('interpolate', help='特征插值生成')
    interp_parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    interp_parser.add_argument('--reference_images', type=str, required=True, help='第一组参考图像')
    interp_parser.add_argument('--reference_images_b', type=str, required=True, help='第二组参考图像')
    
    # 评估命令
    eval_parser = subparsers.add_parser('evaluate', help='评估图像质量')
    eval_parser.add_argument('--real_images', type=str, required=True, help='真实图像目录')
    eval_parser.add_argument('--generated_images', type=str, required=True, help='生成图像目录')
    
    # 测试命令
    test_parser = subparsers.add_parser('test', help='运行测试')
    test_parser.add_argument('--module', type=str, choices=['data', 'model', 'all'], 
                           default='all', help='测试模块')
    
    args, unknown_args = parser.parse_known_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # 根据命令调用相应的脚本
    if args.command == 'train':
        from scripts.train import main as train_main
        # 重新解析参数，传递给训练脚本
        sys.argv = ['scripts/train.py'] + sys.argv[2:]  # 移除 'main.py train'
        train_main()
        
    elif args.command == 'generate':
        from scripts.generate import main as generate_main
        # 设置生成模式
        sys.argv = ['scripts/generate.py', '--mode', 'generate'] + sys.argv[2:]
        generate_main()
        
    elif args.command == 'image2image':
        from scripts.generate import main as generate_main
        sys.argv = ['scripts/generate.py', '--mode', 'image2image'] + sys.argv[2:]
        generate_main()
        
    elif args.command == 'interpolate':
        from scripts.generate import main as generate_main
        sys.argv = ['scripts/generate.py', '--mode', 'interpolate'] + sys.argv[2:]
        generate_main()
        
    elif args.command == 'evaluate':
        from scripts.generate import main as generate_main
        sys.argv = ['scripts/generate.py', '--mode', 'evaluate'] + sys.argv[2:]
        generate_main()
        
    elif args.command == 'test':
        run_tests(args.module)
    
    else:
        print(f"未知命令: {args.command}")
        parser.print_help()


def run_tests(module: str):
    """运行测试"""
    print(f"运行 {module} 模块测试...")
    
    if module == 'data' or module == 'all':
        print("\n=== 测试数据模块 ===")
        try:
            from src.data.isic_dataset import ISICDataset
            from src.data.preprocessing import MedicalImagePreprocessor
            print("✓ 数据模块导入成功")
            
            # 测试预处理器
            processor = MedicalImagePreprocessor()
            print("✓ 预处理器初始化成功")
            
        except Exception as e:
            print(f"✗ 数据模块测试失败: {e}")
    
    if module == 'model' or module == 'all':
        print("\n=== 测试模型模块 ===")
        try:
            from src.models import PanDermFeatureExtractor, VAE, UNet2D
            print("✓ 模型模块导入成功")
            
            # 简单的模型实例化测试
            import torch
            device = torch.device("cpu")  # 使用CPU避免GPU内存问题
            
            extractor = PanDermFeatureExtractor(
                model_name="panderm-large",
                freeze_backbone=True,
                feature_dim=256  # 减小维度
            )
            print("✓ PanDerm特征提取器初始化成功")
            
            vae = VAE(in_channels=3, latent_channels=4, base_channels=32)  # 减小参数
            print("✓ VAE初始化成功")
            
            unet = UNet2D(
                in_channels=4,
                out_channels=4, 
                model_channels=32,  # 减小参数
                context_dim=256
            )
            print("✓ UNet初始化成功")
            
        except Exception as e:
            print(f"✗ 模型模块测试失败: {e}")
    
    print(f"\n{module.capitalize()} 模块测试完成！")


if __name__ == "__main__":
    main()