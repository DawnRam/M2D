#!/usr/bin/env python3
"""
测试WandB可视化功能的脚本
"""

import os
import sys
import torch
import wandb
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.config import Config
from src.training.visualization import WandBVisualizer, create_category_batches, ISIC_CATEGORY_NAMES
from src.models import PanDermFeatureExtractor, VAE, UNet2D
from src.training.diffusion_scheduler import DDIMScheduler


def create_mock_data(batch_size=8, device='cpu'):
    """创建模拟数据"""
    
    # 模拟图像批次
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    targets = torch.randint(0, 5, (batch_size,)).to(device)
    
    batch = {
        'image': images,
        'target': targets,
        'image_id': [f'mock_image_{i}' for i in range(batch_size)],
        'filename': [f'mock_{i}.jpg' for i in range(batch_size)]
    }
    
    # 模拟重构图像
    recon_images = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # 模拟特征
    panderm_features = torch.randn(batch_size, 768).to(device)
    vae_features = torch.randn(batch_size, 512).to(device)
    
    # 模拟损失
    losses = {
        'total_loss': 2.5,
        'diffusion_loss': 1.8,
        'recon_loss': 0.4,
        'repa_loss': 0.2,
        'perceptual_loss': 0.1
    }
    
    return batch, recon_images, panderm_features, vae_features, losses


def create_mock_models(device='cpu'):
    """创建模拟模型"""
    
    panderm_extractor = PanDermFeatureExtractor(
        model_name="panderm-large",
        freeze_backbone=True,
        feature_dim=768
    ).to(device)
    
    vae = VAE(
        in_channels=3,
        latent_channels=4,
        base_channels=32  # 减小参数量用于测试
    ).to(device)
    
    unet = UNet2D(
        in_channels=4,
        out_channels=4,
        model_channels=64,  # 减小参数量用于测试
        context_dim=768,
        use_feature_fusion=True
    ).to(device)
    
    return panderm_extractor, vae, unet


def test_visualization_components():
    """测试可视化组件"""
    
    print("=" * 50)
    print("WandB可视化功能测试")
    print("=" * 50)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 初始化WandB（测试模式）
    wandb.init(
        project="panderm-diffusion-test",
        name="visualization-test",
        mode="offline"  # 离线模式，避免实际上传
    )
    
    try:
        # 1. 创建配置和可视化工具
        config = Config()
        visualizer = WandBVisualizer(config)
        print("✓ WandBVisualizer初始化成功")
        
        # 2. 创建模拟模型
        panderm_extractor, vae, unet = create_mock_models(device)
        visualizer.setup_pipeline(panderm_extractor, vae, unet)
        print("✓ 推理管道设置成功")
        
        # 3. 测试训练图像日志
        print("\n测试训练图像日志...")
        batch, recon_images, panderm_features, vae_features, losses = create_mock_data(8, device)
        
        visualizer.log_training_images(
            epoch=1,
            train_batch=batch,
            recon_images=recon_images,
            max_images=4
        )
        print("✓ 训练图像日志测试完成")
        
        # 4. 测试验证结果日志
        print("\n测试验证结果日志...")
        visualizer.log_validation_results(
            epoch=1,
            val_batch=batch,
            val_losses=losses,
            max_images=4
        )
        print("✓ 验证结果日志测试完成")
        
        # 5. 测试特征分析
        print("\n测试特征分析...")
        visualizer.log_feature_analysis(
            epoch=1,
            panderm_features=panderm_features,
            vae_features=vae_features,
            labels=batch['target']
        )
        print("✓ 特征分析测试完成")
        
        # 6. 测试损失曲线
        print("\n测试损失曲线...")
        visualizer.log_loss_curves(
            epoch=1,
            train_losses=losses,
            val_losses=losses
        )
        print("✓ 损失曲线测试完成")
        
        # 7. 测试学习率日志
        print("\n测试学习率日志...")
        visualizer.log_learning_rate(epoch=1, lr=1e-4)
        print("✓ 学习率日志测试完成")
        
        # 8. 测试梯度统计
        print("\n测试梯度统计...")
        # 创建一些假的梯度
        for param in unet.parameters():
            if param.grad is None:
                param.grad = torch.randn_like(param) * 0.01
            break  # 只测试第一个参数
        
        visualizer.log_model_gradients(epoch=1, model=unet, model_name="unet_test")
        print("✓ 梯度统计测试完成")
        
        # 9. 测试类别批次创建
        print("\n测试类别批次创建...")
        
        class MockDataLoader:
            def __init__(self):
                self.data = []
                for i in range(5):  # 5个批次
                    batch_data = {
                        'image': torch.randn(4, 3, 224, 224).to(device),
                        'target': torch.randint(0, 3, (4,)).to(device)
                    }
                    self.data.append(batch_data)
            
            def __iter__(self):
                return iter(self.data)
        
        mock_loader = MockDataLoader()
        category_batches = create_category_batches(
            mock_loader,
            num_categories=3,
            samples_per_category=5,
            device=device
        )
        
        print(f"✓ 创建了 {len(category_batches)} 个类别批次")
        for cat, images in category_batches.items():
            print(f"  {cat}: {images.shape}")
        
        # 10. 测试diffusion采样类别图像生成（如果有类别批次）
        if category_batches:
            print("\n测试diffusion采样类别图像生成...")
            print("注意：这将通过diffusion pipeline采样生成全新的医学图像")
            try:
                # 减少生成图像数量和推理步数以加快测试
                visualizer.log_generated_images_by_category(
                    epoch=5,
                    reference_batches=category_batches,
                    category_names=ISIC_CATEGORY_NAMES[:len(category_batches)],
                    images_per_category=2,  # 减少数量
                    inference_steps=5      # 减少采样步数
                )
                print("✓ diffusion采样类别图像生成测试完成")
            except Exception as e:
                print(f"⚠ diffusion采样生成测试跳过: {e}")
        
        print("\n" + "=" * 50)
        print("所有测试完成！")
        print("如果启用了在线模式，请检查WandB界面查看可视化结果")
        print("=" * 50)
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        wandb.finish()


def test_category_mapping():
    """测试ISIC类别映射"""
    
    print("\n" + "=" * 30)
    print("ISIC类别名称测试")
    print("=" * 30)
    
    print(f"可用类别数量: {len(ISIC_CATEGORY_NAMES)}")
    for i, category in enumerate(ISIC_CATEGORY_NAMES):
        print(f"  {i}: {category}")
    
    print("✓ ISIC类别名称测试完成")


if __name__ == "__main__":
    # 运行测试
    test_category_mapping()
    test_visualization_components()