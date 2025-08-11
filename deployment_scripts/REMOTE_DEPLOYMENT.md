# 🌐 远程服务器部署指南

## 概述

本指南帮助您将PanDerm-Guided Diffusion项目从Windows开发环境同步到远程GPU服务器进行训练。

## 部署流程

### 第一步：从Windows同步到GitHub

在Windows开发机上运行：

```cmd
sync_to_github.bat
```

这个脚本会：
- ✅ 检查Git环境和配置
- ✅ 初始化Git仓库（如需要）
- ✅ 创建项目提交
- ✅ 配置GitHub远程仓库
- ✅ 推送代码到GitHub
- ✅ 生成远程部署指令

### 第二步：在远程GPU服务器上部署

#### 2.1 连接远程服务器
```bash
ssh username@your-gpu-server.com
```

#### 2.2 克隆项目
```bash
# 使用GitHub同步脚本提供的URL
git clone https://github.com/yourusername/PanDerm-Diffusion.git
cd PanDerm-Diffusion
```

#### 2.3 运行一键部署脚本
```bash
chmod +x deploy_remote.sh
./deploy_remote.sh
```

## 远程部署脚本功能

`deploy_remote.sh` 脚本会自动执行以下操作：

### 环境检查
- 🔍 检查Python 3.8+环境
- 🔍 检查CUDA和GPU支持
- 🔍 显示系统资源信息

### 环境配置  
- 📦 创建Python虚拟环境
- 🔥 根据CUDA版本安装对应PyTorch
- 📚 安装所有项目依赖
- 🧪 验证PyTorch CUDA支持

### 项目设置
- 📁 自动生成模拟ISIC数据集
- ⚙️ 配置模型路径和输出目录
- 🔍 运行完整系统检查

### 训练选项
脚本完成后会提供多种训练选项：

#### 快速测试（推荐首次运行）
```bash
python scripts/train.py --data_root ./data/ISIC --epochs 5 --batch_size 4
```

#### 标准训练
```bash
python scripts/train.py --data_root ./data/ISIC --epochs 50 --batch_size 16
```

#### 完整训练带可视化
```bash
# 先登录WandB
wandb login

# 启动训练
python scripts/train.py \
    --data_root ./data/ISIC \
    --epochs 100 \
    --batch_size 16 \
    --use_wandb \
    --wandb_project "panderm-diffusion"
```

#### 后台长期训练
```bash
nohup python scripts/train.py \
    --data_root ./data/ISIC \
    --epochs 200 \
    --batch_size 16 \
    --use_wandb \
    --wandb_project "panderm-diffusion" \
    > training.log 2>&1 &
```

#### 多GPU训练
```bash
# 首次配置
accelerate config

# 启动多GPU训练
accelerate launch scripts/train.py \
    --data_root ./data/ISIC \
    --epochs 100 \
    --batch_size 32
```

## 监控和管理

### 训练监控
```bash
# 实时查看训练日志
tail -f training.log

# 监控GPU使用情况
watch -n 1 nvidia-smi

# 查看进程状态
ps aux | grep python
```

### WandB可视化
访问 https://wandb.ai 查看训练过程的可视化，包括：
- 📊 损失曲线和学习率变化
- 🖼️ 每个epoch的VAE重构图像对比
- 🎯 每5个epoch的9类皮肤病生成图像展示
- 📈 特征分析和梯度统计

### 检查点管理
```bash
# 查看保存的模型
ls -la checkpoints/

# 测试生成功能
python scripts/generate.py \
    --checkpoint ./checkpoints/best_model.pt \
    --num_samples 20
```

## 常见问题解决

### CUDA内存不足
```bash
# 减小批次大小
--batch_size 8

# 启用梯度累积
--gradient_accumulation_steps 4

# 使用混合精度
--mixed_precision
```

### 依赖安装问题
```bash
# 手动安装PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 网络连接问题
```bash
# 使用国内镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 设置Git代理（如需要）
git config --global http.proxy http://proxy-server:port
```

### WandB登录问题
```bash
# 离线模式
export WANDB_MODE=offline

# 或不使用WandB
python scripts/train.py ... # 不加 --use_wandb 参数
```

## 性能优化建议

### 硬件配置推荐
- **GPU**: RTX 4090/A100/V100 8GB+ 显存
- **CPU**: 8核+ 处理器  
- **内存**: 32GB+ 系统内存
- **存储**: SSD存储（提升数据加载速度）

### 训练参数调优
```bash
# 高性能配置（大显存）
--batch_size 32
--gradient_accumulation_steps 1
--mixed_precision

# 内存友好配置
--batch_size 8  
--gradient_accumulation_steps 4
--mixed_precision
```

### 数据加载优化
```bash
# 增加数据加载进程
--num_workers 8

# 启用数据预获取
--prefetch_factor 2
```

## 结果同步回本地

### 下载训练结果
```bash
# 在本地Windows机器上运行
scp -r username@server:/path/to/PanDerm-Diffusion/checkpoints ./
scp -r username@server:/path/to/PanDerm-Diffusion/generated_samples ./
```

### Git同步更新
```bash
# 在服务器上提交结果
git add checkpoints/ generated_samples/
git commit -m "Add training results and generated samples"
git push origin main

# 在本地拉取更新
git pull origin main
```

## 安全注意事项

1. **API密钥保护**: 确保WandB API key等敏感信息不被提交到Git
2. **服务器访问**: 使用SSH密钥而非密码登录
3. **数据备份**: 定期备份重要的检查点和结果
4. **资源监控**: 监控GPU使用避免资源浪费

## 支持

如果遇到问题：
1. 查看 `training.log` 日志文件
2. 运行 `python check_project.py` 检查系统状态  
3. 检查GPU内存使用 `nvidia-smi`
4. 提交Issue到GitHub项目页面

---

🎉 **祝您训练顺利！** 这个完整的工作流程让您可以轻松地在远程GPU服务器上训练PanDerm-Guided Diffusion模型。