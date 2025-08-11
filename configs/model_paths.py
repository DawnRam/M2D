"""
模型路径配置文件
根据您的实际情况修改这些路径
"""

import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据路径
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "ISIC")

# 模型路径
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# PanDerm模型路径（需要您下载真实模型文件）
PANDERM_MODEL_PATHS = {
    "panderm-base": os.path.join(MODEL_DIR, "panderm", "panderm_base.pth"),
    "panderm-large": os.path.join(MODEL_DIR, "panderm", "panderm_large.pth"),
}

# 如果没有真实的PanDerm模型，将使用ViT替代
USE_VIT_SUBSTITUTE = True

# 输出目录
OUTPUT_DIRS = {
    "checkpoints": os.path.join(PROJECT_ROOT, "checkpoints"),
    "outputs": os.path.join(PROJECT_ROOT, "outputs"), 
    "logs": os.path.join(PROJECT_ROOT, "logs"),
    "generated": os.path.join(PROJECT_ROOT, "generated_images")
}

# 创建必要目录
for dir_path in OUTPUT_DIRS.values():
    os.makedirs(dir_path, exist_ok=True)

print("✓ 模型路径配置完成")
print(f"数据路径: {DATA_ROOT}")
print(f"输出目录: {list(OUTPUT_DIRS.values())}")

# 验证数据路径
if os.path.exists(DATA_ROOT):
    image_dir = os.path.join(DATA_ROOT, "images")
    if os.path.exists(image_dir):
        image_files = [f for f in os.listdir(image_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        print(f"✓ 发现 {len(image_files)} 张图像文件")
    else:
        print(f"⚠ 警告：图像目录不存在: {image_dir}")
else:
    print(f"⚠ 警告：数据根目录不存在: {DATA_ROOT}")
    print("请按照SETUP_GUIDE.md下载和配置ISIC数据集")