#!/bin/bash

# ========================================
# GitHub认证设置脚本
# ========================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "========================================"
echo -e "${BLUE}🔐 GitHub认证配置向导${NC}"
echo "========================================"

echo -e "${BLUE}选择认证方式:${NC}"
echo "1. Personal Access Token (推荐)"
echo "2. SSH密钥"
echo "3. 查看当前配置"
echo

read -p "请选择 (1-3): " choice

case $choice in
    1)
        echo
        echo -e "${BLUE}📝 Personal Access Token 配置${NC}"
        echo "========================================"
        echo
        echo -e "${YELLOW}步骤1: 创建Personal Access Token${NC}"
        echo "1. 访问: https://github.com/settings/tokens"
        echo "2. 点击 'Generate new token' → 'Generate new token (classic)'"
        echo "3. 填写描述: M2D-Project-Access"
        echo "4. 选择过期时间: No expiration 或 1 year"
        echo "5. 勾选权限:"
        echo "   ☑ repo (Full control of repositories)"
        echo "   ☑ workflow (Update GitHub Action workflows)"
        echo "6. 点击 'Generate token' 并复制token"
        echo
        
        read -p "是否已创建token? (y/n): " created
        if [[ "$created" == "y" || "$created" == "Y" ]]; then
            echo
            echo -e "${YELLOW}步骤2: 配置Git凭据存储${NC}"
            
            # 配置凭据存储
            git config --global credential.helper store
            echo -e "${GREEN}✓ 已启用Git凭据存储${NC}"
            
            echo
            echo -e "${YELLOW}步骤3: 测试推送${NC}"
            echo "现在可以运行同步脚本，第一次会提示输入:"
            echo "用户名: DawnRam"
            echo "密码: [粘贴您的Personal Access Token]"
            echo
            echo -e "${GREEN}配置完成！运行 ./sync_to_github.sh 开始推送${NC}"
        fi
        ;;
        
    2)
        echo
        echo -e "${BLUE}🔑 SSH密钥配置${NC}"
        echo "========================================"
        
        # 检查是否已有SSH密钥
        if [ -f ~/.ssh/id_ed25519 ] || [ -f ~/.ssh/id_rsa ]; then
            echo -e "${GREEN}✓ 检测到现有SSH密钥${NC}"
            
            if [ -f ~/.ssh/id_ed25519.pub ]; then
                echo -e "${BLUE}ED25519公钥内容:${NC}"
                cat ~/.ssh/id_ed25519.pub
            elif [ -f ~/.ssh/id_rsa.pub ]; then
                echo -e "${BLUE}RSA公钥内容:${NC}"
                cat ~/.ssh/id_rsa.pub
            fi
        else
            echo -e "${YELLOW}生成新的SSH密钥...${NC}"
            
            # 生成SSH密钥
            if ssh-keygen -t ed25519 -C "cyang5805@gmail.com" -f ~/.ssh/id_ed25519 -N ""; then
                echo -e "${GREEN}✓ SSH密钥生成成功${NC}"
            else
                echo -e "${YELLOW}尝试使用RSA算法...${NC}"
                ssh-keygen -t rsa -b 4096 -C "cyang5805@gmail.com" -f ~/.ssh/id_rsa -N ""
                echo -e "${GREEN}✓ RSA SSH密钥生成成功${NC}"
            fi
            
            # 启动ssh-agent并添加密钥
            eval "$(ssh-agent -s)"
            if [ -f ~/.ssh/id_ed25519 ]; then
                ssh-add ~/.ssh/id_ed25519
                echo -e "${BLUE}公钥内容:${NC}"
                cat ~/.ssh/id_ed25519.pub
            else
                ssh-add ~/.ssh/id_rsa
                echo -e "${BLUE}公钥内容:${NC}"
                cat ~/.ssh/id_rsa.pub
            fi
        fi
        
        echo
        echo -e "${YELLOW}接下来的步骤:${NC}"
        echo "1. 复制上面显示的公钥内容"
        echo "2. 访问: https://github.com/settings/keys"
        echo "3. 点击 'New SSH key'"
        echo "4. 粘贴公钥内容并保存"
        echo "5. 运行以下命令切换到SSH:"
        echo "   git remote set-url origin git@github.com:DawnRam/M2D.git"
        
        read -p "是否已添加SSH密钥到GitHub? (y/n): " added
        if [[ "$added" == "y" || "$added" == "Y" ]]; then
            echo -e "${BLUE}切换到SSH协议...${NC}"
            git remote set-url origin git@github.com:DawnRam/M2D.git
            echo -e "${GREEN}✓ 已切换到SSH协议${NC}"
            
            echo -e "${BLUE}测试SSH连接...${NC}"
            if ssh -T git@github.com; then
                echo -e "${GREEN}✓ SSH连接测试成功！${NC}"
            else
                echo -e "${YELLOW}SSH连接测试失败，但这可能是正常的${NC}"
            fi
        fi
        ;;
        
    3)
        echo
        echo -e "${BLUE}📋 当前Git配置${NC}"
        echo "========================================"
        
        echo -e "${BLUE}用户配置:${NC}"
        echo "用户名: $(git config --global user.name)"
        echo "邮箱: $(git config --global user.email)"
        
        echo
        echo -e "${BLUE}凭据存储:${NC}"
        git config --global credential.helper || echo "未配置"
        
        echo
        echo -e "${BLUE}远程仓库:${NC}"
        if git remote get-url origin 2>/dev/null; then
            remote_url=$(git remote get-url origin)
            if [[ $remote_url == git@github.com* ]]; then
                echo "✓ 使用SSH协议"
            elif [[ $remote_url == https://github.com* ]]; then
                echo "✓ 使用HTTPS协议"
            fi
        else
            echo "未配置远程仓库"
        fi
        
        echo
        echo -e "${BLUE}SSH密钥:${NC}"
        if [ -f ~/.ssh/id_ed25519.pub ]; then
            echo "✓ 检测到ED25519密钥"
        elif [ -f ~/.ssh/id_rsa.pub ]; then
            echo "✓ 检测到RSA密钥"
        else
            echo "未检测到SSH密钥"
        fi
        ;;
        
    *)
        echo -e "${RED}无效选择${NC}"
        exit 1
        ;;
esac

echo
echo -e "${GREEN}🎉 GitHub认证配置向导完成！${NC}"