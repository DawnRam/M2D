# 🚀 M2D 项目部署脚本

本文件夹包含M2D项目的部署相关脚本和文档。

## 📋 文件说明

### 🔄 核心同步脚本
- `smart_sync.sh/.bat` - **智能同步脚本（推荐）**- 自动判断推送或拉取
- `sync_to_github.sh/.bat` - 推送本地代码到GitHub
- `update_from_github.sh/.bat` - 从GitHub拉取最新代码

### 🔐 认证配置
- `setup_github_auth.sh/.bat` - GitHub认证配置向导

### 🚀 部署脚本
- `deploy_remote.sh` - 远程GPU服务器自动部署脚本

### 📚 文档指南
- `COLLABORATION_WORKFLOW.md` - **协作工作流指南**
- `GITHUB_AUTH_SOLUTION.md` - GitHub认证问题解决方案
- `REMOTE_DEPLOYMENT.md` - 详细部署指南
- `GITHUB_SYNC_GUIDE.md` - 完整同步方案说明
- `REMOTE_DEPLOY_COMMAND.txt` - 快速命令参考

## 🔄 使用方法

### 🔐 遇到认证问题？
如果推送时报错认证问题，先运行认证配置脚本：

**Windows:**
```cmd
cd deployment_scripts
setup_github_auth.bat
```

**Linux/Mac:**
```bash
cd deployment_scripts
chmod +x setup_github_auth.sh
./setup_github_auth.sh
```

### 🤖 智能同步（推荐）

**自动判断推送或拉取（最简单）：**

**Windows:**
```cmd
cd deployment_scripts
smart_sync.bat
```

**Linux/Mac:**
```bash
cd deployment_scripts
chmod +x smart_sync.sh
./smart_sync.sh
```

### 📤 手动同步选项

**推送到GitHub:**
```bash
# Linux/Mac
./sync_to_github.sh

# Windows
sync_to_github.bat
```

**从GitHub拉取:**
```bash
# Linux/Mac
./update_from_github.sh

# Windows  
update_from_github.bat
```

### 远程服务器部署
```bash
git clone https://github.com/DawnRam/M2D.git
cd M2D
chmod +x deployment_scripts/deploy_remote.sh
./deployment_scripts/deploy_remote.sh
```

## ⚡ 一键部署命令
```bash
git clone https://github.com/DawnRam/M2D.git && cd M2D && chmod +x deployment_scripts/deploy_remote.sh && ./deployment_scripts/deploy_remote.sh
```

---

**注意**: 这些脚本专门用于从Windows开发环境同步到远程GPU服务器进行训练。