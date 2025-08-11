# M2D - PanDerm-Guided Diffusion for Medical Image Dataset Augmentation

基于PanDerm皮肤镜大模型指导的扩散模型，用于医学图像数据集扩充。

## 🎯 项目特色

- **医学导向**: 利用PanDerm大模型的医学知识指导diffusion生成过程
- **特征融合**: 创新的跨注意力机制融合PanDerm特征和diffusion潜在表示  
- **端到端训练**: 借鉴REPA-E方法，实现VAE和diffusion模型的稳定联合训练
- **WandB可视化**: 每5个epoch自动生成7类皮肤病图像可视化展示
- **Linux优化**: 专为Linux服务器环境优化的训练流程

## 🚀 快速开始

### 环境准备
```bash
# 创建conda环境
conda create -n diff python=3.8
conda activate diff

# 安装依赖
pip install -r requirements.txt
```

### 开始训练
```bash
# 一键训练
chmod +x run_training.sh
./run_training.sh

# 或者直接使用Python
python main.py train
```

## 📋 项目结构

```
M2D/
├── src/                    # 源代码
│   ├── data/               # 数据处理
│   ├── models/             # 模型定义  
│   ├── training/           # 训练相关
│   └── utils/              # 工具函数
├── configs/                # 配置文件
├── scripts/                # 训练和生成脚本

├── main.py                # 主程序入口
└── requirements.txt       # 依赖包列表
```

## 🎨 WandB可视化

训练过程中自动记录：
- 📊 实时损失曲线和学习率变化
- 🖼️ 每个epoch的VAE重构图像对比  
- 🎯 每5个epoch各类皮肤病的diffusion采样生成图像
- 📈 PanDerm特征分析和梯度统计

支持的ISIC皮肤病类别（7类）：
- **AKIEC**: 光化性角化病 (Actinic keratoses)
- **BCC**: 基底细胞癌 (Basal cell carcinoma)  
- **BKL**: 良性角化病 (Benign keratosis-like lesions)
- **DF**: 皮肤纤维瘤 (Dermatofibroma)
- **MEL**: 黑色素瘤 (Melanoma)
- **NV**: 色素痣 (Melanocytic nevi)
- **VASC**: 血管病变 (Vascular lesions)

## ⚙️ 训练选项

```bash
# 激活环境
conda activate diff

# 快速测试（5个epoch）
python scripts/train.py --epochs 5 --batch_size 4

# 完整训练（启用WandB可视化）
python scripts/train.py --epochs 50 --batch_size 16 --use_wandb --wandb_project "m2d-training"

# 后台训练
nohup python scripts/train.py --epochs 50 --batch_size 16 --use_wandb > training.log 2>&1 &
```

## 🖼️ 图像生成

```bash
# 标准生成
python scripts/generate.py --checkpoint ./checkpoints/best_model.pt --num_samples 20

# 图像到图像转换  
python scripts/generate.py --mode image2image --init_images ./input --strength 0.75

# 特征插值
python scripts/generate.py --mode interpolate --reference_images ./ref_a --reference_images_b ./ref_b
```

## 🛠️ 环境要求

- **操作系统**: Linux (推荐Ubuntu 18.04+)
- **Python**: >= 3.8
- **PyTorch**: >= 1.12  
- **CUDA**: >= 11.0
- **显存**: >= 8GB（训练时）
- **Conda**: 用于环境管理

## 📚 技术细节

- **模型架构**: PanDerm特征提取器 + VAE + UNet扩散模型
- **融合策略**: Cross-attention机制融合医学特征
- **损失函数**: 扩散损失 + VAE重构损失 + REPA对齐损失 + 感知损失
- **训练策略**: 端到端联合训练，支持混合精度和多GPU

## 🔧 故障排除

- **CUDA内存不足**: 减小`--batch_size`或启用`--gradient_accumulation_steps`
- **依赖问题**: 根据CUDA版本安装对应PyTorch版本  
- **数据问题**: 运行`python scripts/setup_data.py --all`重新设置
- **训练监控**: 使用`tail -f training.log`查看实时日志

## 📖 数据集和实验目录配置

### 数据集结构
确保ISIC数据集按以下结构组织（每个类别目录包含对应的图像文件）：
```
/nfs/scratch/eechengyang/Data/ISIC/
├── AKIEC/    # 光化性角化病图像 (*.jpg, *.png等)
├── BCC/      # 基底细胞癌图像 (*.jpg, *.png等)
├── BKL/      # 良性角化病图像 (*.jpg, *.png等)
├── DF/       # 皮肤纤维瘤图像 (*.jpg, *.png等)
├── MEL/      # 黑色素瘤图像 (*.jpg, *.png等)
├── NV/       # 色素痣图像 (*.jpg, *.png等)
└── VASC/     # 血管病变图像 (*.jpg, *.png等)
```

**重要说明**: 
- ✅ 系统会自动扫描各类别目录中的图像文件
- ✅ 不需要手动创建metadata.csv文件
- ✅ 支持jpg、jpeg、png、bmp等常见图像格式
- ✅ 每个类别的图像数量可以不同

### 实验目录结构
所有训练产生的文件将保存在独立的实验目录：
```
/nfs/scratch/eechengyang/Code/logs/
├── panderm_diffusion_20231201_143022/    # 实验目录（实验名称_时间戳）
│   ├── checkpoints/      # 模型检查点
│   ├── logs/            # 训练日志
│   ├── outputs/         # 输出文件
│   ├── generated_images/ # 生成的图像
│   ├── cache/           # 缓存文件
│   ├── wandb/           # WandB日志
│   ├── configs/         # 配置文件备份
│   ├── code_backup/     # 代码备份
│   ├── metrics/         # 评估指标
│   ├── visualizations/  # 可视化结果
│   └── experiment_info.json  # 实验信息
├── quick_test_20231201_144523/           # 另一个实验
│   └── ...
└── full_training_20231201_150012/        # 完整训练实验
    └── ...
```

### 实验管理
```bash
# 列出所有实验
python scripts/list_experiments.py list

# 查看实验详情
python scripts/list_experiments.py show experiment_name

# 清理旧实验（保留最近5个）
python scripts/list_experiments.py clean --keep 5
```

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License

---

**免责声明**: 本项目仅用于研究目的，生成的医学图像不应用于实际临床诊断。