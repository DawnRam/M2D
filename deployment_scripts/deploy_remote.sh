#!/bin/bash

# ========================================
# PanDerm-Guided Diffusion 远程服务器部署脚本
# 用于在GPU服务器上部署和运行训练
# ========================================

set -e  # 遇到错误时退出

echo "========================================"
echo "🚀 PanDerm-Guided Diffusion 远程部署"
echo "========================================"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 显示系统信息
echo -e "${BLUE}📊 系统信息:${NC}"
echo "操作系统: $(uname -a)"
echo "Python版本: $(python3 --version 2>/dev/null || echo 'Python未安装')"
echo "GPU信息:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "未检测到NVIDIA GPU"
echo

# 检查Python环境
echo -e "${BLUE}🐍 检查Python环境...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3未安装，请先安装Python 3.8+${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo -e "${GREEN}✓ Python版本: $PYTHON_VERSION${NC}"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo -e "${RED}❌ Python版本过低，需要Python 3.8+${NC}"
    exit 1
fi

# 检查CUDA
echo -e "${BLUE}🔧 检查CUDA环境...${NC}"
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo -e "${GREEN}✓ CUDA版本: $CUDA_VERSION${NC}"
else
    echo -e "${YELLOW}⚠ 未检测到CUDA，将使用CPU模式（训练会很慢）${NC}"
fi

# 创建虚拟环境
echo -e "${BLUE}📦 设置Python虚拟环境...${NC}"
if [ ! -d "panderm_env" ]; then
    echo "创建虚拟环境..."
    python3 -m venv panderm_env
    echo -e "${GREEN}✓ 虚拟环境创建成功${NC}"
else
    echo -e "${GREEN}✓ 虚拟环境已存在${NC}"
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source panderm_env/bin/activate
echo -e "${GREEN}✓ 虚拟环境已激活${NC}"

# 升级pip
echo -e "${BLUE}⬆️ 升级pip...${NC}"
python -m pip install --upgrade pip
echo -e "${GREEN}✓ pip升级完成${NC}"

# 安装PyTorch（根据CUDA版本）
echo -e "${BLUE}🔥 安装PyTorch...${NC}"
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2- | cut -d. -f1,2)
    if [[ "$CUDA_VERSION" == "12."* ]]; then
        echo "安装CUDA 12.x版本的PyTorch..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    elif [[ "$CUDA_VERSION" == "11."* ]]; then
        echo "安装CUDA 11.x版本的PyTorch..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        echo "安装默认CUDA版本的PyTorch..."
        pip install torch torchvision torchaudio
    fi
else
    echo "安装CPU版本的PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi
echo -e "${GREEN}✓ PyTorch安装完成${NC}"

# 安装其他依赖
echo -e "${BLUE}📚 安装项目依赖...${NC}"
pip install -r requirements.txt
echo -e "${GREEN}✓ 依赖安装完成${NC}"

# 验证PyTorch CUDA
echo -e "${BLUE}🧪 验证PyTorch CUDA支持...${NC}"
python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA版本: {torch.version.cuda}')
    print(f'GPU数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('将使用CPU进行训练')
"
echo -e "${GREEN}✓ PyTorch验证完成${NC}"

# 设置数据和模型
echo -e "${BLUE}📁 设置数据和模型路径...${NC}"
python scripts/setup_data.py --all
echo -e "${GREEN}✓ 数据设置完成${NC}"

# 运行系统检查
echo -e "${BLUE}🔍 运行系统完整性检查...${NC}"
python check_project.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 系统检查通过${NC}"
else
    echo -e "${RED}❌ 系统检查失败，请检查输出信息${NC}"
    echo -e "${YELLOW}提示: 可以尝试运行 python scripts/setup_data.py --all 重新设置${NC}"
fi

echo
echo "========================================"
echo -e "${GREEN}🎉 远程服务器部署完成！${NC}"
echo "========================================"
echo

# 显示训练选项
echo -e "${BLUE}🚀 可用的训练选项:${NC}"
echo
echo -e "${YELLOW}1. 快速测试训练 (5个epoch, 推荐首次运行):${NC}"
echo "python scripts/train.py --data_root ./data/ISIC --epochs 5 --batch_size 4 --experiment_name 'quick-test'"
echo

echo -e "${YELLOW}2. 标准训练 (不使用WandB):${NC}"
echo "python scripts/train.py --data_root ./data/ISIC --epochs 50 --batch_size 16"
echo

echo -e "${YELLOW}3. 完整训练 (带WandB可视化):${NC}"
echo "# 首先登录WandB: wandb login"
echo "python scripts/train.py --data_root ./data/ISIC --epochs 100 --batch_size 16 --use_wandb --wandb_project 'panderm-diffusion'"
echo

echo -e "${YELLOW}4. 后台训练 (使用nohup):${NC}"
echo "nohup python scripts/train.py --data_root ./data/ISIC --epochs 100 --batch_size 16 --use_wandb --wandb_project 'panderm-diffusion' > training.log 2>&1 &"
echo

echo -e "${YELLOW}5. GPU多卡训练 (如果有多张GPU):${NC}"
echo "accelerate config  # 首次运行配置"
echo "accelerate launch scripts/train.py --data_root ./data/ISIC --epochs 100 --batch_size 16"
echo

# 自动启动选项
echo -e "${BLUE}🎯 自动启动选项:${NC}"
read -p "是否立即开始快速测试训练？(y/n): " start_training

if [[ "$start_training" == "y" || "$start_training" == "Y" ]]; then
    echo -e "${GREEN}🚀 开始快速测试训练...${NC}"
    python scripts/train.py \
        --data_root ./data/ISIC \
        --epochs 5 \
        --batch_size 4 \
        --learning_rate 1e-4 \
        --experiment_name "remote-quick-test" \
        --save_every 2
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ 快速测试训练完成！${NC}"
        echo "检查点保存在: ./checkpoints/"
        echo "日志保存在: ./logs/"
        echo
        echo -e "${YELLOW}下一步建议:${NC}"
        echo "1. 检查生成的图像质量"
        echo "2. 如果满意，可以运行完整训练"
        echo "3. 使用 'python scripts/generate.py' 测试图像生成"
    else
        echo -e "${RED}❌ 训练过程中出现错误${NC}"
        echo "请检查日志文件并解决问题"
    fi
else
    echo -e "${YELLOW}💡 提示:${NC}"
    echo "- 虚拟环境已激活，可以直接运行上述命令"
    echo "- 要重新激活环境: source panderm_env/bin/activate"
    echo "- 查看训练进度: tail -f training.log"
    echo "- 监控GPU使用: watch -n 1 nvidia-smi"
fi

echo
echo -e "${GREEN}部署脚本执行完成！${NC}"
echo -e "${BLUE}项目路径: $(pwd)${NC}"
echo -e "${BLUE}虚拟环境: panderm_env${NC}"
echo
echo "Happy training! 🎉"