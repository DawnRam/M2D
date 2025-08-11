#!/bin/bash

# ========================================
# M2D GitHub同步脚本 - Linux/Mac版本
# ========================================

set -e  # 遇到错误时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "========================================"
echo -e "${BLUE}M2D GitHub 同步脚本${NC}"
echo "========================================"

# 检查Git是否安装
echo -e "${BLUE}🔍 检查Git环境...${NC}"
if ! command -v git &> /dev/null; then
    echo -e "${RED}❌ Git未安装，请先安装Git${NC}"
    echo "Ubuntu/Debian: sudo apt install git"
    echo "CentOS/RHEL: sudo yum install git"
    echo "macOS: brew install git"
    exit 1
fi

echo -e "${GREEN}✓ Git已安装${NC}"

# 检查Git配置
echo -e "${BLUE}⚙️ 检查Git配置...${NC}"
if ! git config --global user.name &> /dev/null; then
    echo -e "${YELLOW}⚠ Git用户名未配置，设置为项目默认值${NC}"
    git config --global user.name "DawnRam"
    echo -e "${GREEN}✓ 用户名设置为: DawnRam${NC}"
else
    echo -e "${GREEN}✓ 当前用户名: $(git config --global user.name)${NC}"
fi

if ! git config --global user.email &> /dev/null; then
    echo -e "${YELLOW}⚠ Git邮箱未配置，设置为项目默认值${NC}"
    git config --global user.email "cyang5805@gmail.com"
    echo -e "${GREEN}✓ 邮箱设置为: cyang5805@gmail.com${NC}"
else
    echo -e "${GREEN}✓ 当前邮箱: $(git config --global user.email)${NC}"
fi

# 显示当前配置
echo -e "${BLUE}Git配置信息:${NC}"
echo "用户名: $(git config --global user.name)"
echo "邮箱: $(git config --global user.email)"

echo
echo "========================================"
echo -e "${BLUE}📦 准备同步代码到GitHub${NC}"
echo "========================================"

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 检查是否已经是Git仓库
if [ ! -d ".git" ]; then
    echo -e "${YELLOW}初始化Git仓库...${NC}"
    git init -b main
    echo -e "${GREEN}✓ Git仓库初始化完成（默认分支: main）${NC}"
else
    echo -e "${GREEN}✓ Git仓库已存在${NC}"
    
    # 检查当前分支，如果是master则重命名为main
    current_branch=$(git branch --show-current 2>/dev/null || git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
    if [ "$current_branch" = "master" ]; then
        echo -e "${YELLOW}⚠ 当前在master分支，重命名为main...${NC}"
        git branch -m master main
        echo -e "${GREEN}✓ 分支已重命名为main${NC}"
    elif [ "$current_branch" = "main" ]; then
        echo -e "${GREEN}✓ 当前在main分支${NC}"
    elif [ -n "$current_branch" ]; then
        echo -e "${BLUE}ℹ 当前分支: $current_branch${NC}"
    fi
fi

# 添加文件
echo -e "${BLUE}📁 添加文件到Git...${NC}"
git add .
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 文件添加成功${NC}"
else
    echo -e "${RED}❌ 添加文件失败${NC}"
    exit 1
fi

# 检查是否有更改
if ! git diff --cached --quiet; then
    echo -e "${BLUE}📝 创建提交...${NC}"
    git commit -m "Update M2D: PanDerm-Guided Diffusion implementation

Features:
- Complete PanDerm-guided diffusion architecture
- WandB visualization with category image generation every 5 epochs
- REPA-E style end-to-end training
- Medical image preprocessing and augmentation
- Comprehensive evaluation metrics
- Cross-platform deployment scripts
- Mock ISIC dataset generation
- Full documentation and setup guides"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ 提交创建成功${NC}"
    else
        echo -e "${RED}❌ 创建提交失败${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠ 没有新的更改需要提交${NC}"
fi

# 检查远程仓库
echo -e "${BLUE}🔗 检查远程仓库配置...${NC}"
if ! git remote get-url origin &> /dev/null; then
    echo -e "${YELLOW}配置GitHub远程仓库...${NC}"
    repo_url="https://github.com/DawnRam/M2D.git"
    echo "仓库地址: $repo_url"
    
    git remote add origin "$repo_url"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ 远程仓库添加成功${NC}"
    else
        echo -e "${RED}❌ 添加远程仓库失败${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ 远程仓库已配置${NC}"
fi

echo
echo "========================================"
echo -e "${BLUE}🚀 推送代码到GitHub${NC}"
echo "========================================"

# 推送到GitHub
echo -e "${BLUE}⬆️ 推送代码到GitHub...${NC}"

# 确保我们在正确的分支上
current_branch=$(git branch --show-current 2>/dev/null || git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")
echo -e "${BLUE}当前分支: $current_branch${NC}"

# 优先推送到main分支
if git push -u origin "$current_branch" 2>/dev/null; then
    echo -e "${GREEN}✓ 代码推送成功到 $current_branch 分支！${NC}"
elif [ "$current_branch" != "main" ] && git push -u origin main 2>/dev/null; then
    echo -e "${GREEN}✓ 代码推送成功到 main 分支！${NC}"
elif git push -u origin master 2>/dev/null; then
    echo -e "${GREEN}✓ 代码推送成功到 master 分支！${NC}"
else
    echo -e "${RED}❌ 推送失败${NC}"
    echo -e "${YELLOW}可能的原因:${NC}"
    echo "1. 网络连接问题"
    echo "2. GitHub认证问题（需要Personal Access Token）"
    echo "3. 仓库权限问题"
    echo
    echo -e "${YELLOW}解决方案:${NC}"
    echo "1. 检查网络连接"
    echo "2. 配置GitHub Personal Access Token:"
    echo "   - 访问 https://github.com/settings/tokens"
    echo "   - 生成新的token，勾选repo权限"
    echo "   - 使用token作为密码进行Git操作"
    echo "3. 手动推送: git push -u origin main"
    exit 1
fi

echo
echo "========================================"
echo -e "${GREEN}🎉 GitHub同步完成！${NC}"
echo "========================================"
echo
echo -e "${GREEN}您的代码已成功推送到GitHub！${NC}"
echo

# 显示远程部署信息
github_url="https://github.com/DawnRam/M2D.git"
echo -e "${BLUE}下一步 - 远程服务器部署:${NC}"
echo "1. 登录您的远程GPU服务器"
echo "2. 复制以下一键部署命令:"
echo
echo -e "${YELLOW}    git clone $github_url && cd M2D && chmod +x deployment_scripts/deploy_remote.sh && ./deployment_scripts/deploy_remote.sh${NC}"
echo
echo "或者分步执行:"
echo "    git clone $github_url"
echo "    cd M2D"
echo "    chmod +x deployment_scripts/deploy_remote.sh"
echo "    ./deployment_scripts/deploy_remote.sh"
echo

# 创建远程部署信息文件
echo -e "${BLUE}📄 创建部署信息文件...${NC}"
cat > deployment_scripts/REMOTE_DEPLOY.md << EOF
# M2D 远程服务器部署信息

## GitHub仓库地址
$github_url

## 一键部署命令
\`\`\`bash
git clone $github_url && cd M2D && chmod +x deployment_scripts/deploy_remote.sh && ./deployment_scripts/deploy_remote.sh
\`\`\`

## 分步部署命令
\`\`\`bash
git clone $github_url
cd M2D
chmod +x deployment_scripts/deploy_remote.sh
./deployment_scripts/deploy_remote.sh
\`\`\`
EOF

echo -e "${GREEN}✓ 远程部署信息已保存到 deployment_scripts/REMOTE_DEPLOY.md${NC}"
echo
echo -e "${GREEN}🎉 同步脚本执行完成！${NC}"
echo -e "${BLUE}项目已推送至: $github_url${NC}"
echo
echo "Happy coding! 🚀"