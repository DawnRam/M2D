#!/bin/bash

# Linux/Mac脚本 - PanDerm-Guided Diffusion训练

echo "================================"
echo "PanDerm-Guided Diffusion 训练启动"
echo "================================"

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 激活虚拟环境（如果存在）
if [ -f "panderm_env/bin/activate" ]; then
    source panderm_env/bin/activate
    echo "✓ 虚拟环境已激活"
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✓ 虚拟环境已激活"
else
    echo "⚠ 未找到虚拟环境，使用系统Python"
fi

# 检查Python和依赖
echo "检查Python环境..."
if ! python -c "import sys; print(f'Python版本: {sys.version}')" 2>/dev/null; then
    echo "❌ Python未安装或不在PATH中"
    exit 1
fi

# 检查主要依赖
echo "检查关键依赖..."
if ! python -c "import torch, torchvision, transformers; print('✓ 主要依赖检查通过')" 2>/dev/null; then
    echo "❌ 缺少关键依赖，请运行: pip install -r requirements.txt"
    exit 1
fi

# 运行数据设置
echo "设置数据和模型..."
if ! python scripts/setup_data.py --all; then
    echo "❌ 数据设置失败"
    exit 1
fi

# 运行系统测试
echo "运行系统测试..."
if ! python main.py test; then
    echo "❌ 系统测试失败"
    exit 1
fi

# 询问用户是否开始训练
echo ""
read -p "是否开始训练? (y/n): " start_training
if [[ "$start_training" != "y" && "$start_training" != "Y" ]]; then
    echo "训练已取消"
    exit 0
fi

echo "开始训练..."

# 小规模测试训练
echo "开始快速测试训练（5个epoch）..."
if ! python scripts/train.py \
    --data_root ./data/ISIC \
    --batch_size 4 \
    --epochs 5 \
    --learning_rate 1e-4 \
    --panderm_freeze \
    --experiment_name "quick-test"; then
    echo "❌ 快速测试训练失败"
    exit 1
fi

echo ""
echo "✅ 快速测试训练完成！"
echo ""

# 询问是否进行完整训练
read -p "是否进行完整训练? (y/n): " full_training
if [[ "$full_training" != "y" && "$full_training" != "Y" ]]; then
    echo "训练完成"
    exit 0
fi

echo "开始完整训练..."

# 检查是否有WandB配置
if python -c "import wandb; print('WandB可用')" 2>/dev/null; then
    use_wandb="--use_wandb --wandb_project panderm-diffusion"
    echo "✓ 启用WandB记录"
else
    use_wandb=""
    echo "⚠ WandB不可用，跳过在线记录"
fi

# 完整训练
if python scripts/train.py \
    --data_root ./data/ISIC \
    --batch_size 16 \
    --epochs 50 \
    --learning_rate 1e-4 \
    --panderm_freeze \
    --mixed_precision \
    $use_wandb \
    --experiment_name "full-training-v1" \
    --alpha_diffusion 1.0 \
    --beta_recon 0.5 \
    --gamma_repa 0.3 \
    --delta_perceptual 0.2; then
    
    echo ""
    echo "🎉 训练完成！"
    echo ""
    echo "检查结果："
    echo "- 检查点: ./checkpoints/"
    echo "- 日志: ./logs/"
    echo "- 生成图像: ./outputs/"
    if [ -n "$use_wandb" ]; then
        echo "- WandB面板: https://wandb.ai"
    fi
    echo ""
    
    # 询问是否测试生成
    read -p "是否测试图像生成? (y/n): " test_generation
    if [[ "$test_generation" == "y" || "$test_generation" == "Y" ]]; then
        echo "测试图像生成..."
        python scripts/generate.py \
            --checkpoint ./checkpoints/best_model.pt \
            --mode generate \
            --num_samples 10 \
            --output_dir ./generated_samples
        
        echo "✅ 生成测试完成！查看 ./generated_samples/"
    fi
else
    echo "❌ 训练失败"
    exit 1
fi

echo ""
echo "训练流程完成！"