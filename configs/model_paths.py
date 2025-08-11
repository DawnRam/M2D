"""
模型路径配置文件
根据您的实际情况修改这些路径
"""

import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据路径 - 使用您的ISIC数据集路径
DATA_ROOT = "/nfs/scratch/eechengyang/Data/ISIC"

# 模型路径
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# PanDerm模型路径 - 使用您提供的预训练模型路径
# 注意：如果只有一个模型文件，建议只使用一个配置
PANDERM_MODEL_PATHS = {
    "panderm-large": "/nfs/scratch/eechengyang/Code/REPA-E/pretrained/panderm.pth",
    # "panderm-base": "/nfs/scratch/eechengyang/Code/REPA-E/pretrained/panderm_base.pth",  # 如果有base模型，取消注释并修改路径
}

# 使用真实的PanDerm模型
USE_VIT_SUBSTITUTE = False

# 实验输出根目录（独立于代码目录）
EXPERIMENT_ROOT = "/nfs/scratch/eechengyang/Code/logs"

# 默认输出目录（仅用于配置，实际使用时会被实验管理器覆盖）
# 注意：这些路径不会被实际创建，仅作为配置模板
OUTPUT_DIRS = {
    "checkpoints": os.path.join(EXPERIMENT_ROOT, "temp", "checkpoints"),
    "outputs": os.path.join(EXPERIMENT_ROOT, "temp", "outputs"), 
    "logs": os.path.join(EXPERIMENT_ROOT, "temp", "logs"),
    "generated": os.path.join(EXPERIMENT_ROOT, "temp", "generated_images"),
    "cache": os.path.join(EXPERIMENT_ROOT, "temp", "cache"),
    "wandb": os.path.join(EXPERIMENT_ROOT, "temp", "wandb")
}

# 不创建默认目录，由实验管理器负责创建具体实验目录

print("✓ 模型路径配置完成")
print(f"数据路径: {DATA_ROOT}")
print(f"实验根目录: {EXPERIMENT_ROOT}")
print("注意：实际输出目录将由实验管理器创建，格式为 [实验名称_时间戳]")

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