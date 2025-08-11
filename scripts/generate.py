#!/usr/bin/env python3
"""
PanDerm-Guided Diffusion推理脚本
"""

import os
import sys
import argparse
import torch
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.config import Config
from src.training.inference import load_pipeline_from_checkpoint
from src.utils.evaluation import ComprehensiveEvaluator


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="PanDerm-Guided Diffusion图像生成")
    
    # 模型相关
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="模型检查点路径"
    )
    parser.add_argument(
        "--scheduler_type", 
        type=str, 
        default="ddim",
        choices=["ddpm", "ddim"],
        help="采样器类型"
    )
    
    # 生成参数
    parser.add_argument(
        "--mode", 
        type=str, 
        default="generate",
        choices=["generate", "image2image", "interpolate", "evaluate"],
        help="生成模式"
    )
    parser.add_argument(
        "--reference_images", 
        type=str, 
        default=None,
        help="参考图像目录或文件路径"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=8,
        help="生成样本数量"
    )
    parser.add_argument(
        "--num_inference_steps", 
        type=int, 
        default=50,
        help="推理步数"
    )
    parser.add_argument(
        "--guidance_scale", 
        type=float, 
        default=7.5,
        help="引导强度"
    )
    parser.add_argument(
        "--eta", 
        type=float, 
        default=0.0,
        help="DDIM eta参数"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="随机种子"
    )
    
    # 图像到图像参数
    parser.add_argument(
        "--init_images", 
        type=str, 
        default=None,
        help="初始图像路径（用于image2image）"
    )
    parser.add_argument(
        "--strength", 
        type=float, 
        default=0.75,
        help="变形强度（image2image）"
    )
    
    # 插值参数
    parser.add_argument(
        "--reference_images_b", 
        type=str, 
        default=None,
        help="第二组参考图像（用于插值）"
    )
    parser.add_argument(
        "--num_interpolations", 
        type=int, 
        default=8,
        help="插值点数量"
    )
    
    # 评估参数
    parser.add_argument(
        "--real_images", 
        type=str, 
        default=None,
        help="真实图像目录（用于评估）"
    )
    parser.add_argument(
        "--generated_images", 
        type=str, 
        default=None,
        help="生成图像目录（用于评估）"
    )
    
    # 输出
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./generated_images",
        help="输出目录"
    )
    parser.add_argument(
        "--save_format", 
        type=str, 
        default="png",
        choices=["png", "jpg"],
        help="图像保存格式"
    )
    
    return parser.parse_args()


def load_images_from_path(image_path: str, image_size: int = 224) -> torch.Tensor:
    """从路径加载图像"""
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    images = []
    
    if os.path.isfile(image_path):
        # 单个文件
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image)
            images.append(image_tensor)
        except Exception as e:
            print(f"无法加载图像 {image_path}: {e}")
            return None
    
    elif os.path.isdir(image_path):
        # 目录
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in supported_formats:
            image_files.extend([
                f for f in os.listdir(image_path) 
                if f.lower().endswith(ext)
            ])
        
        if not image_files:
            print(f"目录 {image_path} 中没有找到支持的图像文件")
            return None
        
        for filename in sorted(image_files):
            try:
                filepath = os.path.join(image_path, filename)
                image = Image.open(filepath).convert('RGB')
                image_tensor = transform(image)
                images.append(image_tensor)
            except Exception as e:
                print(f"无法加载图像 {filepath}: {e}")
                continue
    
    else:
        print(f"路径 {image_path} 不存在")
        return None
    
    if images:
        return torch.stack(images, dim=0)
    else:
        return None


def generate_images(pipeline, args):
    """生成图像"""
    print("=" * 50)
    print("图像生成模式")
    print("=" * 50)
    
    # 设置随机种子
    generator = torch.Generator().manual_seed(args.seed)
    
    # 加载参考图像
    reference_images = None
    if args.reference_images:
        print(f"加载参考图像: {args.reference_images}")
        reference_images = load_images_from_path(args.reference_images)
        
        if reference_images is None:
            print("无法加载参考图像")
            return
        
        print(f"加载了 {len(reference_images)} 张参考图像")
        reference_images = reference_images.to(pipeline.device)
    
    # 生成图像
    print(f"生成 {args.num_samples} 张图像...")
    
    with torch.no_grad():
        results = pipeline.generate(
            reference_images=reference_images,
            num_samples=args.num_samples,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            eta=args.eta,
            generator=generator,
            return_dict=True
        )
    
    # 保存图像
    os.makedirs(args.output_dir, exist_ok=True)
    pipeline.save_images(
        results['images'],
        args.output_dir,
        prefix="generated",
        format=args.save_format
    )
    
    print(f"图像已保存到: {args.output_dir}")


def image_to_image(pipeline, args):
    """图像到图像生成"""
    print("=" * 50)
    print("图像到图像生成模式")
    print("=" * 50)
    
    if not args.init_images:
        print("错误: image2image模式需要指定--init_images参数")
        return
    
    # 设置随机种子
    generator = torch.Generator().manual_seed(args.seed)
    
    # 加载初始图像
    print(f"加载初始图像: {args.init_images}")
    init_images = load_images_from_path(args.init_images, image_size=256)
    
    if init_images is None:
        print("无法加载初始图像")
        return
    
    print(f"加载了 {len(init_images)} 张初始图像")
    init_images = init_images.to(pipeline.device)
    
    # 加载参考图像（可选）
    reference_images = None
    if args.reference_images:
        print(f"加载参考图像: {args.reference_images}")
        reference_images = load_images_from_path(args.reference_images)
        
        if reference_images is not None:
            reference_images = reference_images.to(pipeline.device)
    
    # 图像到图像生成
    print(f"执行图像转换 (strength={args.strength})...")
    
    with torch.no_grad():
        results = pipeline.image_to_image(
            init_images=init_images,
            reference_images=reference_images,
            strength=args.strength,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            eta=args.eta,
            generator=generator,
            return_dict=True
        )
    
    # 保存图像
    os.makedirs(args.output_dir, exist_ok=True)
    pipeline.save_images(
        results['images'],
        args.output_dir,
        prefix="img2img",
        format=args.save_format
    )
    
    print(f"转换后的图像已保存到: {args.output_dir}")


def interpolate_images(pipeline, args):
    """特征插值生成"""
    print("=" * 50)
    print("特征插值模式")
    print("=" * 50)
    
    if not args.reference_images or not args.reference_images_b:
        print("错误: 插值模式需要指定--reference_images和--reference_images_b参数")
        return
    
    # 设置随机种子
    generator = torch.Generator().manual_seed(args.seed)
    
    # 加载参考图像
    print(f"加载第一组参考图像: {args.reference_images}")
    ref_images_a = load_images_from_path(args.reference_images)
    
    print(f"加载第二组参考图像: {args.reference_images_b}")
    ref_images_b = load_images_from_path(args.reference_images_b)
    
    if ref_images_a is None or ref_images_b is None:
        print("无法加载参考图像")
        return
    
    ref_images_a = ref_images_a.to(pipeline.device)
    ref_images_b = ref_images_b.to(pipeline.device)
    
    # 特征插值
    print(f"生成 {args.num_interpolations} 个插值点...")
    
    with torch.no_grad():
        interpolated_images = pipeline.interpolate(
            reference_images_a=ref_images_a[:1],  # 使用第一张图像
            reference_images_b=ref_images_b[:1],  # 使用第一张图像
            num_interpolations=args.num_interpolations,
            num_inference_steps=args.num_inference_steps,
            eta=args.eta,
            generator=generator
        )
    
    # 保存图像
    os.makedirs(args.output_dir, exist_ok=True)
    pipeline.save_images(
        interpolated_images,
        args.output_dir,
        prefix="interpolated",
        format=args.save_format
    )
    
    print(f"插值图像已保存到: {args.output_dir}")


def evaluate_images(args):
    """评估图像质量"""
    print("=" * 50)
    print("图像质量评估模式")
    print("=" * 50)
    
    if not args.real_images or not args.generated_images:
        print("错误: 评估模式需要指定--real_images和--generated_images参数")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载图像
    print(f"加载真实图像: {args.real_images}")
    real_images = load_images_from_path(args.real_images, image_size=256)
    
    print(f"加载生成图像: {args.generated_images}")
    generated_images = load_images_from_path(args.generated_images, image_size=256)
    
    if real_images is None or generated_images is None:
        print("无法加载图像")
        return
    
    # 转换到[0,1]范围（评估需要）
    if real_images.min() < 0:
        real_images = (real_images + 1.0) / 2.0
    if generated_images.min() < 0:
        generated_images = (generated_images + 1.0) / 2.0
    
    print(f"真实图像数量: {len(real_images)}")
    print(f"生成图像数量: {len(generated_images)}")
    
    # 创建评估器
    evaluator = ComprehensiveEvaluator(device)
    
    # 执行评估
    results = evaluator.evaluate_generation_quality(
        generated_images=generated_images,
        real_images=real_images,
        save_results=True,
        output_dir=args.output_dir
    )
    
    print("\n评估结果:")
    print("=" * 30)
    for key, value in results.items():
        if isinstance(value, dict):
            print(f"{key.upper()}:")
            for k, v in value.items():
                print(f"  {k}: {v:.4f}")
        else:
            print(f"{key.upper()}: {value:.4f}")


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 50)
    print("PanDerm-Guided Diffusion推理")
    print("=" * 50)
    
    # 检查检查点文件
    if not os.path.exists(args.checkpoint):
        print(f"错误: 检查点文件不存在 {args.checkpoint}")
        sys.exit(1)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    try:
        if args.mode == "evaluate":
            # 评估模式不需要加载模型
            evaluate_images(args)
            
        else:
            # 加载推理管道
            print("加载模型...")
            config = Config()  # 使用默认配置
            
            pipeline = load_pipeline_from_checkpoint(
                checkpoint_path=args.checkpoint,
                config=config,
                scheduler_type=args.scheduler_type,
                device=device
            )
            
            print("模型加载完成")
            
            # 根据模式执行相应操作
            if args.mode == "generate":
                generate_images(pipeline, args)
                
            elif args.mode == "image2image":
                image_to_image(pipeline, args)
                
            elif args.mode == "interpolate":
                interpolate_images(pipeline, args)
    
    except Exception as e:
        print(f"推理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("推理完成！")


if __name__ == "__main__":
    main()