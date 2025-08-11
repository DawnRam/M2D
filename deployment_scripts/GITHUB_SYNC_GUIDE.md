# 🔄 GitHub同步与远程GPU服务器部署完整指南

## 解决方案概述

由于Windows系统无法进行GPU训练，本解决方案通过GitHub作为代码同步桥梁，实现从Windows开发环境到远程GPU服务器的无缝部署。

```
Windows开发机 → GitHub → 远程GPU服务器 → 自动部署 → 开始训练
```

## 🚀 一键解决方案

### 在Windows上执行
```cmd
sync_to_github.bat
```

### 在远程GPU服务器上执行  
```bash
# 脚本会提供的一键命令（示例）
git clone https://github.com/yourusername/PanDerm-Diffusion.git && cd PanDerm-Diffusion && chmod +x deploy_remote.sh && ./deploy_remote.sh
```

## 📋 详细步骤

### 第一步：Windows端同步到GitHub

运行 `sync_to_github.bat` 脚本，它会自动：

#### Git环境检查
- ✅ 验证Git是否安装  
- ✅ 配置用户名和邮箱（如未设置）
- ✅ 显示当前Git配置

#### 仓库初始化
- ✅ 创建Git仓库（如不存在）
- ✅ 添加所有项目文件
- ✅ 创建包含完整功能描述的提交

#### GitHub远程配置
- 📝 引导用户在GitHub创建新仓库
- ⚡ 验证仓库URL格式
- 🔗 添加远程origin
- 🚀 推送代码到GitHub

#### 生成部署信息
- 📄 创建 `REMOTE_DEPLOY.md` 包含：
  - GitHub仓库地址
  - 一键部署命令
  - 分步部署命令
- 💡 在终端显示远程部署指令

### 第二步：远程GPU服务器自动部署

运行 `deploy_remote.sh` 脚本，它会自动：

#### 系统环境检查
- 🔍 检查Python 3.8+环境
- 🔍 检测CUDA和GPU支持
- 📊 显示系统资源信息
- ⚡ 验证环境兼容性

#### 环境自动配置
- 📦 创建独立的Python虚拟环境 `panderm_env`
- ⬆️ 升级pip到最新版本
- 🔥 根据CUDA版本智能安装PyTorch：
  - CUDA 12.x → PyTorch CUDA 12.1版本
  - CUDA 11.x → PyTorch CUDA 11.8版本  
  - 无CUDA → PyTorch CPU版本
- 📚 安装所有项目依赖

#### 项目自动设置
- 📁 运行 `setup_data.py` 生成模拟ISIC数据集
- ⚙️ 配置模型路径和输出目录
- 🔍 执行完整系统检查验证

#### 智能训练选项
部署完成后自动提供多种训练选项：

**快速测试（推荐首次运行）**
```bash
python scripts/train.py --data_root ./data/ISIC --epochs 5 --batch_size 4
```

**标准训练**  
```bash  
python scripts/train.py --data_root ./data/ISIC --epochs 50 --batch_size 16
```

**完整训练带WandB可视化**
```bash
wandb login
python scripts/train.py --data_root ./data/ISIC --epochs 100 --batch_size 16 --use_wandb --wandb_project "panderm-diffusion"
```

**后台长期训练**
```bash
nohup python scripts/train.py --data_root ./data/ISIC --epochs 200 --batch_size 16 --use_wandb --wandb_project "panderm-diffusion" > training.log 2>&1 &
```

**多GPU训练**
```bash
accelerate config
accelerate launch scripts/train.py --data_root ./data/ISIC --epochs 100 --batch_size 32
```

## 🛠️ 自动化特性

### Windows端同步脚本特性
- 🔒 **安全验证**: URL格式验证，防止错误配置
- 🔄 **智能检查**: 自动检测Git状态，避免重复操作
- 📝 **详细提示**: 引导用户完成GitHub仓库创建
- 🚨 **错误处理**: 友好的错误信息和解决方案
- 📋 **信息保存**: 生成部署信息文件便于后续使用

### 远程部署脚本特性  
- 🎯 **智能环境检测**: 自动识别CUDA版本并安装对应PyTorch
- 🔧 **依赖管理**: 完整的依赖安装和版本兼容处理
- 📊 **系统验证**: 全面的环境检查确保训练环境可用
- 🚀 **一键启动**: 可选择立即开始快速测试训练
- 🎨 **美观输出**: 彩色终端输出，清晰的状态指示

## 📁 相关文件说明

| 文件 | 用途 |
|------|------|
| `sync_to_github.bat` | Windows同步脚本 |
| `deploy_remote.sh` | 远程服务器部署脚本 |
| `REMOTE_DEPLOYMENT.md` | 详细部署指南 |
| `REMOTE_DEPLOY_COMMAND.txt` | 快速命令参考 |
| `GITHUB_SYNC_GUIDE.md` | 本文件，完整方案说明 |
| `REMOTE_DEPLOY.md` | 自动生成的用户专属部署信息 |

## 🌟 解决方案优势

### 开发体验
- **一键同步**: Windows端一键推送到GitHub
- **一键部署**: 远程服务器一键环境配置
- **无需手动配置**: 自动处理环境、依赖、数据设置
- **智能错误处理**: 友好的错误提示和解决方案

### 训练效率
- **快速启动**: 从零到开始训练仅需几分钟
- **环境隔离**: 独立虚拟环境避免冲突
- **多种选项**: 从快速测试到完整训练的多种模式
- **监控便利**: 集成WandB可视化和日志管理

### 可维护性
- **版本控制**: 通过Git管理代码版本
- **模块化设计**: 各个脚本功能独立，易于维护
- **文档完整**: 详细的使用说明和故障排除指南
- **扩展友好**: 易于添加新功能或适配不同环境

## 🔧 自定义配置

### 修改训练参数
在远程服务器上可以通过命令行参数自定义：

```bash
python scripts/train.py \
    --data_root ./data/ISIC \
    --batch_size 32 \
    --epochs 200 \
    --learning_rate 1e-4 \
    --fusion_type cross_attention \
    --alpha_diffusion 1.0 \
    --beta_recon 0.5 \
    --gamma_repa 0.3 \
    --delta_perceptual 0.2 \
    --mixed_precision \
    --use_wandb \
    --wandb_project "panderm-diffusion"
```

### 使用真实数据
将真实ISIC数据集放置到远程服务器：
```bash
# 上传数据到服务器
scp -r ./real_isic_data username@server:/path/to/PanDerm-Diffusion/data/ISIC/

# 使用真实数据训练
python scripts/train.py --data_root ./data/ISIC
```

## 🚨 故障排除

### Windows端常见问题
- **Git未安装**: 脚本会提示下载安装Git
- **网络问题**: 检查GitHub连接，可能需要VPN
- **认证失败**: 使用Personal Access Token作为密码

### 远程服务器常见问题
- **Python版本过低**: 脚本会自动检查并提示
- **CUDA不兼容**: 自动安装对应版本的PyTorch  
- **内存不足**: 减小batch_size或使用梯度累积
- **网络连接**: 使用国内镜像源安装依赖

## 🎯 预期效果

使用本解决方案，您可以：
- ⏱️ **5分钟内**从Windows开发环境同步到GitHub
- ⏱️ **10分钟内**在远程GPU服务器完成环境配置  
- ⏱️ **15分钟内**开始第一次训练测试
- 📊 **实时监控**通过WandB查看训练进度
- 🎨 **每5个epoch**自动生成9类皮肤病图像可视化

## 💡 使用建议

1. **首次使用**: 建议先运行5个epoch的快速测试
2. **监控训练**: 使用 `tail -f training.log` 实时查看训练日志
3. **资源监控**: 使用 `watch -n 1 nvidia-smi` 监控GPU使用
4. **定期备份**: 定期下载重要的检查点文件到本地
5. **版本管理**: 重要改动及时提交到GitHub保存

---

🎉 **这套完整的解决方案让您可以轻松实现跨平台的PanDerm-Guided Diffusion训练！** 从Windows开发到远程GPU训练，一切都自动化处理。