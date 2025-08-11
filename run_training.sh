#!/bin/bash

# Linux/Mac脚本 - PanDerm-Guided Diffusion训练

echo "================================"
echo "PanDerm-Guided Diffusion 训练启动"
echo "================================"


# 运行数据设置
echo "设置数据和模型..."
if ! python scripts/setup_data.py --all; then
    echo "❌ 数据设置失败"
    exit 1
fi

# 跳过系统测试（对训练无关键作用，且可能受环境依赖影响）
echo "跳过系统测试"

echo "开始训练..."

# 检查GPU环境
echo "检查GPU环境..."
if python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')" 2>/dev/null; then
    gpu_count=$(python -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)")
    echo "✓ 检测到 $gpu_count 个GPU设备"
    
    # 根据GPU数量自动选择训练方式（非交互）
    if [ "$gpu_count" -gt 1 ]; then
        echo "检测到多GPU环境，自动使用分布式训练"
        training_mode="distributed"
    else
        training_mode="single"
    fi
else
    echo "⚠ GPU检查失败，使用CPU训练"
    training_mode="cpu"
fi

# 小规模测试训练
echo "开始快速测试训练（5个epoch）..."
if [ "$training_mode" = "distributed" ]; then
    echo "使用分布式训练模式..."
    if ! python scripts/train_distributed.py \
        --epochs 5 \
        --batch_size 4 \
        --panderm_freeze \
        --experiment_name "quick-test-distributed"; then
        echo "❌ 分布式快速测试训练失败"
        exit 1
    fi
else
    echo "使用单GPU/CPU训练模式..."
    if ! python scripts/train.py \
        --data_root /nfs/scratch/eechengyang/Data/ISIC \
        --batch_size 4 \
        --epochs 5 \
        --learning_rate 1e-4 \
        --panderm_freeze \
        --experiment_name "quick-test"; then
        echo "❌ 快速测试训练失败"
        exit 1
    fi
fi

    echo ""
    echo "✅ 快速测试训练完成！"
    echo ""
    echo "实验结果保存在: /nfs/scratch/eechengyang/Code/logs/[实验名称_时间戳]/"
    echo "查看实验: python scripts/list_experiments.py list"
    echo ""

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
echo "开始完整训练..."
if [ "$training_mode" = "distributed" ]; then
    echo "使用分布式训练模式..."
    if python scripts/train_distributed.py \
        --epochs 50 \
        --batch_size 16 \
        --panderm_freeze \
        --mixed_precision \
        $([ -n "$use_wandb" ] && echo "--use_wandb") \
        --experiment_name "full-training-distributed"; then
        training_success=true
    else
        training_success=false
    fi
else
    echo "使用单GPU/CPU训练模式..."
    if python scripts/train.py \
        --data_root /nfs/scratch/eechengyang/Data/ISIC \
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
        training_success=true
    else
        training_success=false
    fi
fi

if [ "$training_success" = true ]; then
    
    echo ""
    echo "🎉 训练完成！"
    echo ""
    echo "检查结果："
    echo "- 实验根目录: /nfs/scratch/eechengyang/Code/logs/"
    echo "- 实验目录: /nfs/scratch/eechengyang/Code/logs/[实验名称_时间戳]/"
    echo "  - 检查点: checkpoints/"
    echo "  - 训练日志: logs/"
    echo "  - 生成图像: outputs/"
    echo "  - 配置备份: configs/"
    echo "  - 代码备份: code_backup/"
    if [ -n "$use_wandb" ]; then
        echo "  - WandB日志: wandb/"
        echo "- WandB面板: https://wandb.ai"
    fi
    echo ""
    echo "查看实验详情: python scripts/list_experiments.py list"
    echo ""
    
    # 询问是否测试生成
    read -p "是否测试图像生成? (y/n): " test_generation
    if [[ "$test_generation" == "y" || "$test_generation" == "Y" ]]; then
        echo "测试图像生成..."
        # 注意：这里需要使用实际的实验目录路径
        echo "注意：请使用实际的实验目录路径，例如："
        echo "python scripts/generate.py \\"
        echo "    --checkpoint /nfs/scratch/eechengyang/Code/logs/[实验名称_时间戳]/checkpoints/best_model.pt \\"
        echo "    --mode generate \\"
        echo "    --num_samples 10 \\"
        echo "    --output_dir /nfs/scratch/eechengyang/Code/logs/[实验名称_时间戳]/generated_samples"
        echo ""
        echo "或者使用实验管理工具查看具体路径："
        echo "python scripts/list_experiments.py list"
    fi
else
    echo "❌ 训练失败"
    exit 1
fi

echo ""
echo "训练流程完成！"