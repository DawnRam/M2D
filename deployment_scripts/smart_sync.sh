#!/bin/bash

# ========================================
# M2D 智能同步脚本 - 自动判断推送或拉取
# ========================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo "========================================"
echo -e "${CYAN}🔄 M2D 智能同步脚本${NC}"
echo "========================================"

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 检查Git仓库
if [ ! -d ".git" ]; then
    echo -e "${YELLOW}⚠ 当前目录不是Git仓库，将初始化并推送${NC}"
    exec ./deployment_scripts/sync_to_github.sh
    exit $?
fi

# 检查网络连接
echo -e "${BLUE}🌐 检查GitHub连接...${NC}"
if ! git ls-remote origin &>/dev/null; then
    echo -e "${RED}❌ 无法连接到GitHub仓库${NC}"
    echo "请检查网络连接和认证配置"
    exit 1
fi
echo -e "${GREEN}✓ GitHub连接正常${NC}"

# 获取本地和远程状态
current_branch=$(git branch --show-current 2>/dev/null || git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")
echo -e "${BLUE}当前分支: $current_branch${NC}"

# 获取远程信息
git fetch origin -q

# 获取提交哈希
local_commit=$(git rev-parse HEAD 2>/dev/null || echo "")
remote_commit=$(git rev-parse origin/$current_branch 2>/dev/null || git rev-parse origin/main 2>/dev/null || echo "")

# 检查本地是否有未提交的更改
has_local_changes=false
if ! git diff-index --quiet HEAD -- 2>/dev/null; then
    has_local_changes=true
fi

# 检查是否有未跟踪的文件
has_untracked_files=false
if [ -n "$(git ls-files --others --exclude-standard)" ]; then
    has_untracked_files=true
fi

echo
echo -e "${PURPLE}📊 仓库状态分析:${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -z "$remote_commit" ]; then
    echo -e "${YELLOW}📤 远程仓库为空或无法访问${NC}"
    echo -e "${BLUE}建议操作: 推送本地代码到GitHub${NC}"
    action="push"
elif [ -z "$local_commit" ]; then
    echo -e "${YELLOW}📥 本地仓库为空${NC}"
    echo -e "${BLUE}建议操作: 从GitHub拉取代码${NC}"
    action="pull"
elif [ "$local_commit" = "$remote_commit" ]; then
    if [ "$has_local_changes" = true ] || [ "$has_untracked_files" = true ]; then
        echo -e "${YELLOW}📝 本地有未提交的更改${NC}"
        echo -e "${BLUE}建议操作: 推送本地更改到GitHub${NC}"
        action="push"
    else
        echo -e "${GREEN}✅ 本地和远程代码同步${NC}"
        echo -e "${BLUE}建议操作: 无需同步${NC}"
        action="none"
    fi
else
    # 检查分叉情况
    merge_base=$(git merge-base HEAD origin/$current_branch 2>/dev/null || git merge-base HEAD origin/main 2>/dev/null || echo "")
    
    if [ "$merge_base" = "$local_commit" ]; then
        echo -e "${YELLOW}📥 远程有新提交${NC}"
        echo -e "${BLUE}建议操作: 从GitHub拉取更新${NC}"
        action="pull"
    elif [ "$merge_base" = "$remote_commit" ]; then
        echo -e "${YELLOW}📤 本地有新提交${NC}"
        echo -e "${BLUE}建议操作: 推送到GitHub${NC}"
        action="push"
    else
        echo -e "${YELLOW}🔀 本地和远程都有新提交（已分叉）${NC}"
        echo -e "${BLUE}建议操作: 先拉取远程更新再推送${NC}"
        action="sync"
    fi
fi

# 显示详细状态
if [ "$has_local_changes" = true ]; then
    echo -e "${YELLOW}• 有未提交的修改${NC}"
fi

if [ "$has_untracked_files" = true ]; then
    echo -e "${YELLOW}• 有未跟踪的文件${NC}"
fi

# 显示提交差异
if [ -n "$local_commit" ] && [ -n "$remote_commit" ] && [ "$local_commit" != "$remote_commit" ]; then
    ahead_count=$(git rev-list --count origin/$current_branch..HEAD 2>/dev/null || git rev-list --count origin/main..HEAD 2>/dev/null || echo "0")
    behind_count=$(git rev-list --count HEAD..origin/$current_branch 2>/dev/null || git rev-list --count HEAD..origin/main 2>/dev/null || echo "0")
    
    if [ "$ahead_count" -gt 0 ]; then
        echo -e "${CYAN}• 本地领先 $ahead_count 个提交${NC}"
    fi
    
    if [ "$behind_count" -gt 0 ]; then
        echo -e "${CYAN}• 本地落后 $behind_count 个提交${NC}"
    fi
fi

echo

# 根据分析结果执行操作
case $action in
    "push")
        echo -e "${BLUE}🚀 执行推送操作...${NC}"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        exec ./deployment_scripts/sync_to_github.sh
        ;;
    "pull")
        echo -e "${BLUE}📥 执行拉取操作...${NC}"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        exec ./deployment_scripts/update_from_github.sh
        ;;
    "sync")
        echo -e "${BLUE}🔄 执行双向同步操作...${NC}"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "步骤1: 先拉取远程更新"
        ./deployment_scripts/update_from_github.sh
        if [ $? -eq 0 ]; then
            echo
            echo "步骤2: 推送本地更改"
            ./deployment_scripts/sync_to_github.sh
        else
            echo -e "${RED}❌ 拉取失败，请解决冲突后手动推送${NC}"
            exit 1
        fi
        ;;
    "none")
        echo -e "${GREEN}✨ 恭喜！代码已经同步${NC}"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo
        echo -e "${BLUE}可用操作:${NC}"
        echo "• 强制检查更新: ./deployment_scripts/update_from_github.sh"
        echo "• 推送新更改: ./deployment_scripts/sync_to_github.sh"
        echo "• 查看状态: git status"
        echo "• 查看历史: git log --oneline -10"
        ;;
esac

echo
echo -e "${GREEN}🎉 智能同步完成！${NC}"