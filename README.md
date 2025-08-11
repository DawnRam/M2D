# M2D - PanDerm-Guided Diffusion for Medical Image Dataset Augmentation

基于PanDerm皮肤镜大模型指导的扩散模型，用于医学图像数据集扩充。

## 🎯 项目特色

- **医学导向**: 利用PanDerm大模型的医学知识指导diffusion生成过程
- **特征融合**: 创新的跨注意力机制融合PanDerm特征和diffusion潜在表示  
- **端到端训练**: 借鉴REPA-E方法，实现VAE和diffusion模型的稳定联合训练
- **WandB可视化**: 每5个epoch自动生成9类皮肤病图像可视化展示
- **一键部署**: 完整的自动化部署脚本，支持从Windows到远程GPU服务器

## 🚀 快速开始

### 本地训练（Linux/Mac）
```bash
# 设置环境
chmod +x run_training.sh
./run_training.sh
```

### 远程GPU服务器训练  
```bash
# 一键部署命令
git clone https://github.com/DawnRam/M2D.git && cd M2D && chmod +x deployment_scripts/deploy_remote.sh && ./deployment_scripts/deploy_remote.sh
```

### Windows用户
```cmd
# 同步到GitHub后在远程服务器部署
cd deployment_scripts
sync_to_github.bat
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
├── deployment_scripts/     # 部署脚本
├── main.py                # 主程序入口
└── requirements.txt       # 依赖包列表
```

## 🎨 WandB可视化

训练过程中自动记录：
- 📊 实时损失曲线和学习率变化
- 🖼️ 每个epoch的VAE重构图像对比  
- 🎯 每5个epoch各类皮肤病的diffusion采样生成图像
- 📈 PanDerm特征分析和梯度统计

支持的ISIC皮肤病类别：黑色素瘤、色素痣、基底细胞癌、光化性角化病、良性角化病、皮肤纤维瘤、血管病变、鳞状细胞癌、其他

## ⚙️ 训练选项

```bash
# 快速测试（5个epoch）
python scripts/train.py --data_root ./data/ISIC --epochs 5 --batch_size 4

# 完整训练（启用WandB可视化）
python scripts/train.py --data_root ./data/ISIC --epochs 100 --batch_size 16 --use_wandb --wandb_project "m2d-training"

# 后台训练
nohup python scripts/train.py --data_root ./data/ISIC --epochs 100 --batch_size 16 --use_wandb > training.log 2>&1 &
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

- Python >= 3.8
- PyTorch >= 1.12  
- CUDA >= 11.0（推荐）
- 显存 >= 8GB（训练时）

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

## 📖 更多信息

- 详细部署指南: `deployment_scripts/README.md`
- 完整同步方案: `deployment_scripts/GITHUB_SYNC_GUIDE.md`
- 系统检查: `python check_project.py`

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License

---

**免责声明**: 本项目仅用于研究目的，生成的医学图像不应用于实际临床诊断。