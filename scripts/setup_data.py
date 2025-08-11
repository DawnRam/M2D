#!/usr/bin/env python3
"""
数据集下载和配置脚本
"""

import os
import sys
import requests
import zipfile
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.model_paths import DATA_ROOT, PROJECT_ROOT


def download_file(url, filepath, description="下载中"):
    """下载文件并显示进度条"""
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as file, tqdm(
        desc=description,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)


def download_sample_isic_data():
    """下载示例ISIC数据"""
    
    print("正在下载示例ISIC数据...")
    
    # 创建数据目录
    images_dir = os.path.join(DATA_ROOT, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # 示例图像URL（这里使用占位符，实际需要替换为真实的ISIC数据源）
    sample_urls = [
        # 这些是示例URL，实际使用时需要替换为真实的ISIC数据下载链接
        # "https://isic-archive.com/api/v1/image/download/ISIC_0000001",
        # "https://isic-archive.com/api/v1/image/download/ISIC_0000002",
    ]
    
    # 如果没有真实URL，创建模拟数据
    if not sample_urls:
        print("创建模拟ISIC数据用于测试...")
        create_mock_isic_data()
        return
    
    # 下载真实数据
    for i, url in enumerate(sample_urls):
        filename = f"ISIC_{i:07d}.jpg"
        filepath = os.path.join(images_dir, filename)
        
        try:
            download_file(url, filepath, f"下载 {filename}")
        except Exception as e:
            print(f"下载 {filename} 失败: {e}")


def create_mock_isic_data():
    """创建模拟ISIC数据用于测试"""
    
    print("创建模拟ISIC数据...")
    
    images_dir = os.path.join(DATA_ROOT, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    try:
        from PIL import Image, ImageDraw
        import numpy as np
        
        # 创建模拟皮肤镜图像
        categories = [
            "melanoma", "nevus", "basal_cell", "keratosis", 
            "benign", "dermatofibroma", "vascular", "squamous", "other"
        ]
        
        images_per_category = 50  # 每个类别50张图像
        image_size = (224, 224)
        
        metadata_list = []
        
        for cat_idx, category in enumerate(categories):
            print(f"创建 {category} 类别图像...")
            
            for img_idx in range(images_per_category):
                # 创建基础图像
                img = Image.new('RGB', image_size, color='white')
                draw = ImageDraw.Draw(img)
                
                # 添加模拟皮肤纹理
                np.random.seed(cat_idx * 100 + img_idx)  # 确保可重现
                
                # 绘制背景色（模拟皮肤色调）
                skin_colors = [
                    (245, 222, 179),  # 浅肤色
                    (241, 194, 125),  # 中等肤色
                    (198, 134, 66),   # 深肤色
                ]
                bg_color = skin_colors[cat_idx % len(skin_colors)]
                img = Image.new('RGB', image_size, color=bg_color)
                draw = ImageDraw.Draw(img)
                
                # 添加随机斑点/病变
                for _ in range(np.random.randint(1, 5)):
                    x = np.random.randint(0, image_size[0])
                    y = np.random.randint(0, image_size[1])
                    radius = np.random.randint(5, 30)
                    
                    # 不同类别的不同颜色
                    if category == "melanoma":
                        color = (139, 69, 19)  # 深棕色
                    elif category == "nevus":
                        color = (160, 82, 45)  # 鞍褐色
                    elif category == "basal_cell":
                        color = (255, 182, 193)  # 浅粉色
                    else:
                        color = (np.random.randint(100, 200), 
                                np.random.randint(50, 150), 
                                np.random.randint(50, 150))
                    
                    draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)
                
                # 保存图像
                filename = f"ISIC_{cat_idx:02d}{img_idx:03d}.jpg"
                filepath = os.path.join(images_dir, filename)
                img.save(filepath, quality=85)
                
                # 记录元数据
                metadata_list.append({
                    'image_id': os.path.splitext(filename)[0],
                    'filename': filename,
                    'target': cat_idx,
                    'category': category
                })
        
        # 保存元数据
        metadata_df = pd.DataFrame(metadata_list)
        metadata_path = os.path.join(DATA_ROOT, "metadata.csv")
        metadata_df.to_csv(metadata_path, index=False)
        
        print(f"✓ 创建了 {len(metadata_list)} 张模拟图像")
        print(f"✓ 元数据已保存到: {metadata_path}")
        
    except ImportError as e:
        print(f"创建模拟数据需要PIL: {e}")
        print("请运行: pip install Pillow")


def download_pretrained_weights():
    """下载预训练权重（如果可用）"""
    
    print("配置预训练模型...")
    
    # 创建模型目录
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(os.path.join(models_dir, "panderm"), exist_ok=True)
    os.makedirs(os.path.join(models_dir, "pretrained"), exist_ok=True)
    
    print("由于PanDerm模型需要特殊权限，将使用ViT作为替代")
    print("系统会在训练时自动下载必要的预训练ViT权重")


def verify_setup():
    """验证设置"""
    
    print("\n" + "="*50)
    print("验证设置...")
    
    # 检查数据目录
    images_dir = os.path.join(DATA_ROOT, "images")
    if os.path.exists(images_dir):
        image_files = [f for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"✓ 图像目录: {len(image_files)} 张图像")
        
        if len(image_files) > 100:
            print("✓ 图像数量充足")
        else:
            print("⚠ 图像数量较少，建议增加更多数据")
    else:
        print(f"✗ 图像目录不存在: {images_dir}")
        return False
    
    # 检查元数据
    metadata_path = os.path.join(DATA_ROOT, "metadata.csv")
    if os.path.exists(metadata_path):
        try:
            metadata = pd.read_csv(metadata_path)
            print(f"✓ 元数据文件: {len(metadata)} 条记录")
            
            if 'target' in metadata.columns:
                print(f"✓ 类别分布: {metadata['target'].value_counts().to_dict()}")
        except Exception as e:
            print(f"⚠ 元数据读取错误: {e}")
    else:
        print("⚠ 未找到元数据文件")
    
    # 检查输出目录
    from configs.model_paths import OUTPUT_DIRS
    for name, path in OUTPUT_DIRS.items():
        if os.path.exists(path):
            print(f"✓ {name}目录: {path}")
        else:
            print(f"✗ {name}目录不存在: {path}")
    
    print("="*50)
    return True


def main():
    parser = argparse.ArgumentParser(description="设置ISIC数据集和模型")
    parser.add_argument("--download-sample", action="store_true", help="下载示例数据")
    parser.add_argument("--create-mock", action="store_true", help="创建模拟数据") 
    parser.add_argument("--download-models", action="store_true", help="下载预训练模型")
    parser.add_argument("--verify", action="store_true", help="验证设置")
    parser.add_argument("--all", action="store_true", help="执行所有设置步骤")
    
    args = parser.parse_args()
    
    if args.all:
        args.create_mock = True
        args.download_models = True
        args.verify = True
    
    if not any([args.download_sample, args.create_mock, args.download_models, args.verify]):
        args.all = True
        args.create_mock = True
        args.download_models = True
        args.verify = True
    
    print("PanDerm-Guided Diffusion 数据设置")
    print("="*50)
    
    try:
        if args.download_sample:
            download_sample_isic_data()
        
        if args.create_mock:
            create_mock_isic_data()
        
        if args.download_models:
            download_pretrained_weights()
        
        if args.verify:
            success = verify_setup()
            if success:
                print("\n🎉 设置完成！现在可以开始训练了")
                print("\n下一步:")
                print("1. python main.py test  # 运行系统测试")
                print("2. python scripts/train.py --help  # 查看训练参数")
                print("3. python scripts/train.py --data_root ./data/ISIC --epochs 10  # 开始训练")
            else:
                print("\n❌ 设置不完整，请检查错误信息")
        
    except KeyboardInterrupt:
        print("\n⚠ 用户中断")
    except Exception as e:
        print(f"\n❌ 设置过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()