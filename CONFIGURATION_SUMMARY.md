# M2D项目配置总结

## 项目概述
M2D是一个基于**REPA-E（Representation Alignment）框架**的医学图像生成项目，通过引入PanDerm皮肤镜大模型进行特征对齐，实现ISIC皮肤病数据集的智能扩充。

### 核心技术架构
1. **REPA-E框架**：通过表示对齐损失（REPA Loss）实现VAE潜在特征与PanDerm医学特征的对齐
2. **PanDerm特征提取**：利用预训练的PanDerm大模型提取医学语义特征
3. **扩散模型生成**：使用UNet2D作为去噪网络，生成高质量医学图像
4. **特征融合机制**：通过交叉注意力机制融合PanDerm特征和扩散模型的潜在表示

## 已完成的配置修改

### 1. 数据路径配置
- **原路径**: `./data/ISIC`
- **新路径**: `/nfs/scratch/eechengyang/Data/ISIC`
- **修改文件**:
  - `configs/model_paths.py`
  - `scripts/train.py`
  - `run_training.sh`

### 2. PanDerm模型路径配置
- **模型路径**: `/nfs/scratch/eechengyang/Code/REPA-E/pretrained/panderm.pth`
- **修改内容**:
  - 设置`USE_VIT_SUBSTITUTE = False`，使用真实的PanDerm模型
  - 更新`PanDermFeatureExtractor`类，支持加载真实的预训练权重
- **修改文件**:
  - `configs/model_paths.py`
  - `src/models/panderm_extractor.py`

### 3. 多类别数据集支持
创建了新的`ISICMultiClassDataset`类，支持ISIC的7个类别：

| 类别代码 | 中文名称 | 英文名称 |
|---------|---------|----------|
| AKIEC | 光化性角化病 | Actinic keratoses |
| BCC | 基底细胞癌 | Basal cell carcinoma |
| BKL | 良性角化病 | Benign keratosis-like lesions |
| DF | 皮肤纤维瘤 | Dermatofibroma |
| MEL | 黑色素瘤 | Melanoma |
| NV | 色素痣 | Melanocytic nevi |
| VASC | 血管病变 | Vascular lesions |

**新增文件**:
- `src/data/isic_multiclass_dataset.py` - 多类别数据集加载器

**修改文件**:
- `src/data/__init__.py` - 导出新的数据集类
- `src/training/trainer.py` - 使用新的数据加载器
- `src/training/visualization.py` - 更新类别名称

### 4. 关键特性
- **类别平衡采样**：通过过采样实现类别平衡
- **数据增强**：针对医学图像的专用增强策略
- **缓存机制**：支持数据集信息缓存，加速加载
- **智能权重加载**：兼容不同格式的PanDerm模型checkpoint

## 使用说明

### 1. 验证配置
```bash
# 运行配置测试脚本
python test_configuration.py
```

### 2. 快速测试训练
```bash
# 5个epoch的快速测试
python scripts/train.py \
    --data_root /nfs/scratch/eechengyang/Data/ISIC \
    --batch_size 4 \
    --epochs 5 \
    --panderm_freeze \
    --experiment_name "quick-test"
```

### 3. 完整训练
```bash
# 使用自动化脚本
./run_training.sh

# 或者手动运行完整训练
python scripts/train.py \
    --data_root /nfs/scratch/eechengyang/Data/ISIC \
    --batch_size 16 \
    --epochs 50 \
    --learning_rate 1e-4 \
    --panderm_freeze \
    --mixed_precision \
    --use_wandb \
    --alpha_diffusion 1.0 \
    --beta_recon 0.5 \
    --gamma_repa 0.3 \
    --delta_perceptual 0.2
```

### 4. 生成图像
```bash
# 使用训练好的模型生成图像
python scripts/generate.py \
    --checkpoint ./checkpoints/best_model.pt \
    --mode generate \
    --num_samples 10 \
    --output_dir ./generated_samples
```

## REPA-E框架详解

### 损失函数组成
1. **Diffusion Loss** (α=1.0): 扩散模型的去噪损失
2. **Reconstruction Loss** (β=0.5): VAE重构损失
3. **REPA Loss** (γ=0.3): 特征对齐损失，确保VAE特征与PanDerm特征对齐
4. **Perceptual Loss** (δ=0.2): 感知损失，提高生成图像质量

### 特征对齐机制
- **目标**：让VAE编码器学习到的潜在特征与PanDerm提取的医学特征对齐
- **方法**：使用余弦相似度或KL散度计算对齐损失
- **效果**：生成的图像包含更准确的医学特征

## 项目优势
1. **医学知识引导**：利用PanDerm大模型的医学先验知识
2. **高质量生成**：通过REPA-E框架实现稳定的特征对齐
3. **类别均衡**：支持不平衡数据集的均衡采样
4. **可扩展性**：易于适配其他医学图像数据集

## 注意事项
1. 确保ISIC数据集的7个类别目录结构正确
2. PanDerm模型文件必须存在且格式正确
3. 建议使用GPU进行训练，CPU训练速度会很慢
4. 首次运行会创建数据集缓存，可能需要较长时间

## 后续优化建议
1. 调整损失函数权重以获得最佳生成效果
2. 尝试不同的特征融合策略（cross_attention, adaptive等）
3. 根据具体需求调整数据增强策略
4. 使用WandB监控训练过程并调整超参数