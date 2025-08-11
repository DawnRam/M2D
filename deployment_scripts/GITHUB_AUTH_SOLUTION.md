# 🔐 GitHub推送认证问题解决方案

## ⚡ 快速解决方案

### 最简单的方法（推荐）：

1. **创建Personal Access Token**：
   - 访问：https://github.com/settings/tokens
   - 点击 "Generate new token" → "Generate new token (classic)"
   - 填写描述：`M2D-Project-Access`
   - 勾选权限：`repo` 和 `workflow`
   - 点击生成并**立即复制保存token**

2. **配置Git凭据存储**：
   ```bash
   git config --global credential.helper store
   ```

3. **运行同步脚本**，第一次推送时输入：
   - 用户名：`DawnRam`
   - 密码：`[粘贴您的Personal Access Token]`

## 🚀 自动化解决方案

### Linux/Mac用户：
```bash
cd deployment_scripts
chmod +x setup_github_auth.sh
./setup_github_auth.sh
```

### Windows用户：
```cmd
cd deployment_scripts
setup_github_auth.bat
```

## 🔧 手动解决方案

### 方案A：使用Token URL（立即可用）
```bash
# 临时推送命令
git push https://DawnRam:[YOUR_TOKEN]@github.com/DawnRam/M2D.git main

# 或永久设置remote URL
git remote set-url origin https://DawnRam:[YOUR_TOKEN]@github.com/DawnRam/M2D.git
```

### 方案B：使用GitHub CLI（最安全）
```bash
# 安装GitHub CLI
winget install GitHub.cli  # Windows
brew install gh           # Mac
# Linux: 参考 https://github.com/cli/cli#installation

# 登录并推送
gh auth login
git push origin main
```

### 方案C：SSH密钥（长期推荐）
```bash
# 生成SSH密钥
ssh-keygen -t ed25519 -C "cyang5805@gmail.com"

# 复制公钥并添加到GitHub
cat ~/.ssh/id_ed25519.pub

# 修改remote URL
git remote set-url origin git@github.com:DawnRam/M2D.git
```

## 📋 常见问题

### Q: Token创建后在哪里输入？
A: 运行推送命令时，Git会提示输入用户名和密码：
- 用户名：`DawnRam`  
- 密码：粘贴您的Personal Access Token

### Q: Token忘记了怎么办？
A: Token只显示一次，忘记了需要重新生成：
1. 访问 https://github.com/settings/tokens
2. 删除旧token，创建新token

### Q: 推送还是失败怎么办？
A: 检查以下几点：
1. Token权限是否包含 `repo`
2. 用户名是否正确：`DawnRam`
3. 仓库地址是否正确：`https://github.com/DawnRam/M2D.git`

### Q: 如何避免每次都输入Token？
A: 启用Git凭据存储：
```bash
git config --global credential.helper store
```
输入一次后，Git会自动保存凭据。

## 🎯 推荐流程

1. **立即解决**（30秒）：
   ```bash
   git push https://DawnRam:[YOUR_TOKEN]@github.com/DawnRam/M2D.git main
   ```

2. **永久配置**（2分钟）：
   ```bash
   git config --global credential.helper store
   git remote set-url origin https://github.com/DawnRam/M2D.git
   git push origin main  # 输入用户名和token
   ```

3. **后续使用**：
   ```bash
   ./sync_to_github.sh  # Linux/Mac
   sync_to_github.bat   # Windows
   ```

## 💡 安全提示

- ✅ Token具有过期时间，建议设置1年
- ✅ 不要在代码中硬编码token
- ✅ 使用credential helper安全存储
- ✅ 定期更新和轮换token
- ❌ 不要在公开地方分享token

---

选择任一方案即可解决认证问题，推荐使用Personal Access Token方案！