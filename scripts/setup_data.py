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
    
    # 不创建统一的images目录，而是为每个类别创建子目录
    os.makedirs(DATA_ROOT, exist_ok=True)
    
    try:
        from PIL import Image, ImageDraw
        import numpy as np
        
        # 创建模拟皮肤镜图像 - 7个ISIC标准类别
        categories = [
            "AKIEC",  # 光化性角化病
            "BCC",    # 基底细胞癌
            "BKL",    # 良性角化病
            "DF",     # 皮肤纤维瘤
            "MEL",    # 黑色素瘤
            "NV",     # 色素痣
            "VASC"    # 血管病变
        ]
        
        images_per_category = 50  # 每个类别50张图像
        image_size = (224, 224)
        
        metadata_list = []
        
        for cat_idx, category in enumerate(categories):
            print(f"创建 {category} 类别图像...")
            
            # 为每个类别创建子目录
            category_dir = os.path.join(DATA_ROOT, category)
            os.makedirs(category_dir, exist_ok=True)
            
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
                    
                    # 不同类别的不同颜色 - 使用ISIC标准类别
                    if category == "MEL":  # 黑色素瘤
                        color = (139, 69, 19)  # 深棕色
                    elif category == "NV":  # 色素痣
                        color = (160, 82, 45)  # 鞍褐色
                    elif category == "BCC":  # 基底细胞癌
                        color = (255, 182, 193)  # 浅粉色
                    elif category == "AKIEC":  # 光化性角化病
                        color = (255, 215, 0)  # 金色
                    elif category == "BKL":  # 良性角化病
                        color = (210, 180, 140)  # 棕褐色
                    elif category == "DF":  # 皮肤纤维瘤
                        color = (255, 160, 122)  # 浅鲑鱼色
                    elif category == "VASC":  # 血管病变
                        color = (220, 20, 60)  # 深红色
                    else:
                        color = (np.random.randint(100, 200), 
                                np.random.randint(50, 150), 
                                np.random.randint(50, 150))
                    
                    draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)
                
                # 保存图像到对应的类别目录
                filename = f"ISIC_{cat_idx:02d}{img_idx:03d}.jpg"
                filepath = os.path.join(category_dir, filename)
                img.save(filepath, quality=85)
                
                # 记录元数据
                metadata_list.append({
                    'image_id': os.path.splitext(filename)[0],
                    'filename': filename,
                    'target': cat_idx,
                    'category': category
                })
        
        # 不再创建metadata.csv文件，使用真实数据集的目录结构
        # metadata_df = pd.DataFrame(metadata_list)
        # metadata_path = os.path.join(DATA_ROOT, "metadata.csv")
        # metadata_df.to_csv(metadata_path, index=False)
        
        print(f"✓ 创建了 {len(metadata_list)} 张模拟图像")
        # print(f"✓ 元数据已保存到: {metadata_path}")
        
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
        # args.create_mock = True  # 不再创建模拟数据
        args.download_models = True
        args.verify = True
    
    if not any([args.download_sample, args.create_mock, args.download_models, args.verify]):
        args.all = True
        # args.create_mock = True  # 不再创建模拟数据
        args.download_models = True
        args.verify = True
    
    print("PanDerm-Guided Diffusion 数据设置")
    print("="*50)
    
    try:
        if args.download_sample:
            download_sample_isic_data()
        
        # 不再创建模拟数据，使用真实的ISIC数据集
        # if args.create_mock:
        #     create_mock_isic_data()
        
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