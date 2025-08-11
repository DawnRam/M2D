#!/bin/bash

# ========================================
# M2D GitHub更新脚本 - Linux/Mac版本
# 从GitHub拉取最新代码并智能合并
# ========================================

set -e  # 遇到错误时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo "========================================"
echo -e "${BLUE}📥 M2D GitHub 更新脚本${NC}"
echo "========================================"

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 检查Git仓库
if [ ! -d ".git" ]; then
    echo -e "${RED}❌ 当前目录不是Git仓库${NC}"
    echo "请先运行同步脚本初始化仓库"
    exit 1
fi

echo -e "${GREEN}✓ Git仓库已存在${NC}"

# 检查网络连接
echo -e "${BLUE}🌐 检查网络连接...${NC}"
if ! git ls-remote origin &>/dev/null; then
    echo -e "${RED}❌ 无法连接到GitHub仓库${NC}"
    echo "请检查网络连接和认证配置"
    exit 1
fi
echo -e "${GREEN}✓ GitHub连接正常${NC}"

# 获取当前状态
current_branch=$(git branch --show-current 2>/dev/null || git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")
echo -e "${BLUE}当前分支: $current_branch${NC}"

# 检查本地是否有未提交的更改
echo -e "${BLUE}🔍 检查本地更改...${NC}"
if ! git diff-index --quiet HEAD -- 2>/dev/null; then
    echo -e "${YELLOW}⚠ 检测到本地未提交的更改:${NC}"
    git status --porcelain
    echo
    
    echo -e "${BLUE}选择处理方式:${NC}"
    echo "1. 暂存本地更改后更新 (推荐)"
    echo "2. 提交本地更改后更新"
    echo "3. 丢弃本地更改并强制更新"
    echo "4. 退出脚本"
    echo
    
    read -p "请选择 (1-4): " choice
    
    case $choice in
        1)
            echo -e "${YELLOW}暂存本地更改...${NC}"
            git stash push -m "Auto stash before update $(date '+%Y-%m-%d %H:%M:%S')"
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}✓ 本地更改已暂存${NC}"
                stashed=true
            else
                echo -e "${RED}❌ 暂存失败${NC}"
                exit 1
            fi
            ;;
        2)
            echo -e "${BLUE}提交本地更改...${NC}"
            git add .
            read -p "请输入提交信息: " commit_msg
            if [ -z "$commit_msg" ]; then
                commit_msg="Local changes before GitHub update $(date '+%Y-%m-%d %H:%M:%S')"
            fi
            git commit -m "$commit_msg"
            echo -e "${GREEN}✓ 本地更改已提交${NC}"
            ;;
        3)
            echo -e "${RED}⚠ 警告: 这将丢弃所有本地更改！${NC}"
            read -p "确认要丢弃本地更改吗? (输入 'yes' 确认): " confirm
            if [ "$confirm" = "yes" ]; then
                git reset --hard HEAD
                git clean -fd
                echo -e "${YELLOW}✓ 本地更改已丢弃${NC}"
            else
                echo "操作取消"
                exit 1
            fi
            ;;
        4)
            echo "退出脚本"
            exit 0
            ;;
        *)
            echo -e "${RED}无效选择${NC}"
            exit 1
            ;;
    esac
else
    echo -e "${GREEN}✓ 工作区是干净的${NC}"
    stashed=false
fi

# 获取远程更新信息
echo -e "${BLUE}📡 获取远程仓库信息...${NC}"
git fetch origin

# 检查是否有远程更新
local_commit=$(git rev-parse HEAD)
remote_commit=$(git rev-parse origin/$current_branch 2>/dev/null || git rev-parse origin/main 2>/dev/null || echo "")

if [ -z "$remote_commit" ]; then
    echo -e "${RED}❌ 无法获取远程分支信息${NC}"
    exit 1
fi

if [ "$local_commit" = "$remote_commit" ]; then
    echo -e "${GREEN}✓ 本地代码已是最新版本${NC}"
    
    # 如果之前有暂存的更改，询问是否恢复
    if [ "$stashed" = true ]; then
        echo
        read -p "是否恢复之前暂存的更改? (y/n): " restore_stash
        if [[ "$restore_stash" == "y" || "$restore_stash" == "Y" ]]; then
            git stash pop
            echo -e "${GREEN}✓ 暂存的更改已恢复${NC}"
        fi
    fi
    exit 0
fi

# 显示即将更新的内容
echo
echo -e "${PURPLE}📋 远程更新概览:${NC}"
git log --oneline --graph HEAD..origin/$current_branch 2>/dev/null || git log --oneline --graph HEAD..origin/main
echo

# 确认更新
read -p "是否继续更新? (y/n): " confirm_update
if [[ "$confirm_update" != "y" && "$confirm_update" != "Y" ]]; then
    echo "更新取消"
    exit 0
fi

# 执行更新
echo -e "${BLUE}📥 拉取远程更新...${NC}"

# 尝试快进合并
if git merge --ff-only origin/$current_branch 2>/dev/null; then
    echo -e "${GREEN}✓ 快进更新成功${NC}"
    merge_success=true
elif git merge --ff-only origin/main 2>/dev/null; then
    echo -e "${GREEN}✓ 快进更新成功 (从main分支)${NC}"
    merge_success=true
else
    # 需要三方合并
    echo -e "${YELLOW}⚠ 需要进行合并操作${NC}"
    echo "正在尝试自动合并..."
    
    if git merge origin/$current_branch -m "Merge remote changes from origin/$current_branch"; then
        echo -e "${GREEN}✓ 自动合并成功${NC}"
        merge_success=true
    elif git merge origin/main -m "Merge remote changes from origin/main"; then
        echo -e "${GREEN}✓ 自动合并成功 (从main分支)${NC}"
        merge_success=true
    else
        echo -e "${RED}❌ 自动合并失败，存在冲突${NC}"
        echo
        echo -e "${YELLOW}冲突文件:${NC}"
        git diff --name-only --diff-filter=U
        echo
        echo -e "${BLUE}解决冲突的步骤:${NC}"
        echo "1. 手动编辑冲突文件"
        echo "2. 运行: git add <已解决的文件>"
        echo "3. 运行: git commit"
        echo "4. 或者运行: git merge --abort 取消合并"
        merge_success=false
    fi
fi

# 如果合并成功且有暂存的更改，询问是否恢复
if [ "$merge_success" = true ] && [ "$stashed" = true ]; then
    echo
    read -p "是否恢复之前暂存的更改? (y/n): " restore_stash
    if [[ "$restore_stash" == "y" || "$restore_stash" == "Y" ]]; then
        if git stash pop; then
            echo -e "${GREEN}✓ 暂存的更改已恢复${NC}"
            
            # 检查是否有冲突
            if ! git diff-index --quiet HEAD -- 2>/dev/null; then
                echo -e "${YELLOW}⚠ 暂存的更改与远程更新可能有冲突${NC}"
                echo "请检查并解决任何冲突"
            fi
        else
            echo -e "${YELLOW}⚠ 暂存的更改恢复时有冲突，请手动处理${NC}"
        fi
    else
        echo -e "${BLUE}ℹ 暂存的更改仍保存在stash中，可以稍后用 'git stash pop' 恢复${NC}"
    fi
fi

# 显示更新结果
echo
echo "========================================"
if [ "$merge_success" = true ]; then
    echo -e "${GREEN}🎉 GitHub更新完成！${NC}"
    echo "========================================"
    echo
    echo -e "${BLUE}📊 更新统计:${NC}"
    git diff --stat HEAD@{1} HEAD 2>/dev/null || echo "无法显示统计信息"
    echo
    echo -e "${BLUE}📝 最新提交:${NC}"
    git log --oneline -5
else
    echo -e "${YELLOW}⚠ 更新未完成${NC}"
    echo "========================================"
    echo "请解决合并冲突后手动完成更新"
fi

echo
echo -e "${CYAN}提示:${NC}"
echo "- 查看更新内容: git log --oneline -10"
echo "- 查看文件更改: git diff HEAD~1"
echo "- 如需回滚更新: git reset --hard HEAD~1"
echo "- 查看暂存列表: git stash list"
echo
echo "Happy coding! 🚀"