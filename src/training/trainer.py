import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, Any
import logging
from tqdm import tqdm
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
import json
import numpy as np

from ..models import PanDermFeatureExtractor, VAE, UNet2D
from ..data import create_dataloaders
from .diffusion_scheduler import DDPMScheduler, DDIMScheduler
from .losses import CombinedLoss
from .visualization import WandBVisualizer, create_category_batches, ISIC_CATEGORY_NAMES
from configs.config import Config


class PanDermDiffusionTrainer:
    """PanDerm-Guided Diffusion训练器"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # 设置GPU环境
        self._setup_gpu_environment()
        
        # 初始化accelerator
        accelerator_kwargs = {
            'mixed_precision': 'fp16' if config.mixed_precision else 'no',
            'gradient_accumulation_steps': config.gradient_accumulation_steps,
            'log_with': "wandb" if wandb.run else None,
        }
        
        # 分布式训练的特殊设置
        if config.distributed:
            print("✓ 配置分布式训练accelerator")
            # 让accelerate自动处理设备分配，不手动指定
            accelerator_kwargs.update({
                'device_placement': True,  # 自动设备放置
                'split_batches': False,    # 不分割批次
            })
        
        self.accelerator = Accelerator(**accelerator_kwargs)
        
        # 设置随机种子
        set_seed(config.seed)
        
        # 设置日志
        self._setup_logging()
        
        # 初始化模型
        self.panderm_extractor = None
        self.vae = None
        self.unet = None
        self.scheduler = None
        
        # 训练状态
        self.current_epoch = 0
        self.current_step = 0
        self.best_loss = float('inf')
        
        # 数据加载器
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # 优化器和调度器
        self.optimizer = None
        self.lr_scheduler = None
        
        # 损失函数
        self.criterion = None
        
        # 可视化工具
        self.visualizer = None
        self.category_batches = None
    
    def _setup_gpu_environment(self):
        """设置GPU环境"""
        import os
        
        # 检测可用GPU数量
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"✓ 检测到 {num_gpus} 个GPU设备")
            
            # 对于分布式训练，不设置CUDA_VISIBLE_DEVICES，让accelerate管理
            if self.config.distributed:
                print("✓ 分布式训练模式：由accelerate管理GPU分配")
                # 移除CUDA_VISIBLE_DEVICES设置，避免冲突
                if "CUDA_VISIBLE_DEVICES" in os.environ:
                    print(f"  移除CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
                    del os.environ["CUDA_VISIBLE_DEVICES"]
            else:
                # 单GPU训练时才设置CUDA_VISIBLE_DEVICES
                if self.config.gpu_ids and self.config.gpu_ids != "auto":
                    os.environ["CUDA_VISIBLE_DEVICES"] = self.config.gpu_ids
                    print(f"✓ 单GPU训练：设置可见GPU设备: {self.config.gpu_ids}")
            
            # 自动设置进程数
            if self.config.num_processes == -1:
                self.config.num_processes = num_gpus
                print(f"✓ 自动设置进程数为: {num_gpus}")
            
            # 打印GPU信息
            for i in range(min(num_gpus, 4)):  # 只显示前4个GPU信息，避免输出过长
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            if num_gpus > 4:
                print(f"  ... 和其他 {num_gpus-4} 个GPU")
        else:
            print("⚠ 未检测到CUDA设备，将使用CPU训练")
            self.config.distributed = False
            self.config.num_processes = 1
        
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    os.path.join(self.config.log_dir, 'training.log')
                ),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 输出目录由实验管理器创建，这里不再重复创建
        # os.makedirs(self.config.output_dir, exist_ok=True)
        # os.makedirs(self.config.checkpoint_dir, exist_ok=True) 
        # os.makedirs(self.config.log_dir, exist_ok=True)
    
    def setup_models(self):
        """初始化模型"""
        self.logger.info("初始化模型...")
        
        # PanDerm特征提取器
        self.panderm_extractor = PanDermFeatureExtractor(
            model_name=self.config.model.panderm_model,
            freeze_backbone=self.config.model.panderm_freeze,
            feature_dim=self.config.model.feature_dim
        )
        
        # VAE
        self.vae = VAE(
            in_channels=3,
            latent_channels=self.config.model.vae_channels,
            base_channels=128
        )
        
        # UNet
        self.unet = UNet2D(
            in_channels=self.config.model.vae_channels,
            out_channels=self.config.model.vae_channels,
            model_channels=self.config.model.unet_channels,
            time_embed_dim=self.config.model.time_embed_dim,
            attention_resolutions=(4, 2, 1),
            channel_mult=(1, 2, 4, 4),
            num_heads=self.config.model.attention_heads,
            context_dim=self.config.model.feature_dim,
            use_feature_fusion=True,
            fusion_type=self.config.model.fusion_type
        )
        
        # Diffusion调度器
        self.scheduler = DDPMScheduler(
            num_train_timesteps=self.config.training.num_diffusion_steps,
            beta_schedule=self.config.training.noise_schedule
        )
        
        self.logger.info("模型初始化完成")
        self.logger.info(f"PanDerm参数量: {sum(p.numel() for p in self.panderm_extractor.parameters()):,}")
        self.logger.info(f"VAE参数量: {sum(p.numel() for p in self.vae.parameters()):,}")
        self.logger.info(f"UNet参数量: {sum(p.numel() for p in self.unet.parameters()):,}")
    
    def setup_data(self):
        """初始化数据加载器"""
        self.logger.info("初始化数据加载器...")
        
        # 使用多类别数据集加载器
        from ..data import create_isic_dataloaders
        
        # 获取缓存目录（从配置或实验目录）
        cache_dir = None
        if hasattr(self.config, 'cache_dir') and self.config.cache_dir:
            cache_dir = self.config.cache_dir
        elif hasattr(self.config, 'log_dir') and self.config.log_dir:
            # 使用日志目录的父目录作为缓存目录
            import os
            cache_dir = os.path.join(os.path.dirname(self.config.log_dir), "cache")
        
        self.train_loader, self.val_loader, self.test_loader = create_isic_dataloaders(
            data_root=self.config.data.data_root,
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers,
            image_size=self.config.data.image_size,
            augmentation=self.config.data.augmentation,
            balance_classes=True,  # 启用类别平衡
            cache_dir=cache_dir
        )
        
        self.logger.info(f"训练集大小: {len(self.train_loader.dataset)}")
        self.logger.info(f"验证集大小: {len(self.val_loader.dataset)}")
        self.logger.info(f"测试集大小: {len(self.test_loader.dataset)}")
        
        # 创建类别批次用于可视化
        if wandb.run:
            self.logger.info("创建可视化用的类别批次...")
            self.category_batches = create_category_batches(
                self.val_loader,
                num_categories=min(len(ISIC_CATEGORY_NAMES), 8),  # 最多8个类别
                samples_per_category=10,
                device=self.accelerator.device
            )
            self.logger.info(f"创建了 {len(self.category_batches)} 个类别批次进行可视化")
    
    def setup_optimizer(self):
        """初始化优化器和学习率调度器"""
        self.logger.info("初始化优化器...")
        
        # 分别为不同模块设置不同的学习率
        param_groups = []
        
        # PanDerm特征提取器（如果不冻结）
        if not self.config.model.panderm_freeze:
            param_groups.append({
                'params': self.panderm_extractor.parameters(),
                'lr': self.config.training.learning_rate * 0.1,  # 更小的学习率
                'name': 'panderm'
            })
        
        # VAE
        param_groups.append({
            'params': self.vae.parameters(),
            'lr': self.config.training.learning_rate,
            'name': 'vae'
        })
        
        # UNet
        param_groups.append({
            'params': self.unet.parameters(),
            'lr': self.config.training.learning_rate,
            'name': 'unet'
        })
        
        self.optimizer = optim.AdamW(
            param_groups,
            weight_decay=self.config.training.weight_decay
        )
        
        # 学习率调度器
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.training.epochs,
            eta_min=1e-6
        )
        
        # 损失函数
        self.criterion = CombinedLoss(
            alpha_diffusion=self.config.training.alpha_diffusion,
            beta_recon=self.config.training.beta_reconstruction,
            gamma_repa=self.config.training.gamma_alignment,
            delta_perceptual=self.config.training.delta_perceptual
        )
    
    def setup_accelerator(self):
        """设置Accelerator"""
        self.logger.info("设置Accelerator...")
        
        # 准备模型、优化器和数据加载器
        (
            self.panderm_extractor,
            self.vae,
            self.unet,
            self.optimizer,
            self.train_loader,
            self.val_loader
        ) = self.accelerator.prepare(
            self.panderm_extractor,
            self.vae,
            self.unet,
            self.optimizer,
            self.train_loader,
            self.val_loader
        )
        
        # 损失函数到设备
        self.criterion = self.criterion.to(self.accelerator.device)
        self.scheduler.betas = self.scheduler.betas.to(self.accelerator.device)
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.accelerator.device)
        
        # 初始化可视化工具
        if wandb.run:
            self.visualizer = WandBVisualizer(self.config)
            self.visualizer.setup_pipeline(
                self.panderm_extractor,
                self.vae, 
                self.unet
            )
            self.logger.info("WandB可视化工具初始化完成")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """单个训练步骤"""
        
        images = batch['image']  # [B, 3, H, W]
        batch_size = images.shape[0]
        
        # 1. PanDerm特征提取
        with torch.no_grad() if self.config.model.panderm_freeze else torch.enable_grad():
            panderm_features = self.panderm_extractor(images)['global']  # [B, D]
        
        # 2. VAE编码
        with self.accelerator.accumulate(self.vae):
            mu, logvar = self.vae.encode(images)
            latents = self.vae.reparameterize(mu, logvar)  # [B, 4, H', W']
            
            # VAE重构（用于重构损失）
            recon_images = self.vae.decode(latents)
        
        # 3. 扩散过程
        with self.accelerator.accumulate(self.unet):
            # 随机时间步
            timesteps = torch.randint(
                0, self.scheduler.num_train_timesteps, (batch_size,),
                device=self.accelerator.device
            )
            
            # 添加噪声
            noise = torch.randn_like(latents)
            noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
            
            # UNet预测
            model_output = self.unet(
                noisy_latents,
                timesteps,
                panderm_features=panderm_features
            )['sample']
        
        # 4. 计算损失
        loss_dict = self.criterion(
            model_output=model_output,
            noise_target=noise,
            recon_images=recon_images,
            target_images=images,
            vae_features=mu,  # 使用VAE的mu作为特征
            panderm_features=panderm_features,
            timesteps=timesteps
        )
        
        # 5. 反向传播
        self.accelerator.backward(loss_dict['total_loss'])
        
        if self.accelerator.sync_gradients:
            # 梯度裁剪
            self.accelerator.clip_grad_norm_(self.unet.parameters(), 1.0)
            self.accelerator.clip_grad_norm_(self.vae.parameters(), 1.0)
            if not self.config.model.panderm_freeze:
                self.accelerator.clip_grad_norm_(self.panderm_extractor.parameters(), 1.0)
        
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # 转换为标量值用于日志
        scalar_losses = {}
        for key, value in loss_dict.items():
            if torch.is_tensor(value):
                scalar_losses[key] = value.item()
            else:
                scalar_losses[key] = value
        
        return scalar_losses, {
            'images': images,
            'recon_images': recon_images,
            'panderm_features': panderm_features,
            'vae_features': mu,
            'generated_noise': model_output
        }
    
    def validate(self) -> Dict[str, float]:
        """验证"""
        self.unet.eval()
        self.vae.eval()
        if not self.config.model.panderm_freeze:
            self.panderm_extractor.eval()
        
        total_loss = 0.0
        num_batches = 0
        loss_components = {}
        
        val_batch_for_vis = None
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.val_loader, desc="验证中", leave=False)):
                batch_loss = self.val_step(batch)
                
                # 保存第一个批次用于可视化
                if i == 0 and self.visualizer is not None:
                    val_batch_for_vis = batch
                
                total_loss += batch_loss['total_loss']
                num_batches += 1
                
                # 累计各项损失
                for key, value in batch_loss.items():
                    if key not in loss_components:
                        loss_components[key] = 0.0
                    loss_components[key] += value
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        for key in loss_components:
            loss_components[key] /= num_batches
        
        # 恢复训练模式
        self.unet.train()
        self.vae.train()
        if not self.config.model.panderm_freeze:
            self.panderm_extractor.train()
        
        return {'val_avg_loss': avg_loss, **loss_components}, val_batch_for_vis
    
    def val_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """单个验证步骤"""
        
        images = batch['image']
        batch_size = images.shape[0]
        
        # PanDerm特征提取
        panderm_features = self.panderm_extractor(images)['global']
        
        # VAE编码
        mu, logvar = self.vae.encode(images)
        latents = self.vae.reparameterize(mu, logvar)
        recon_images = self.vae.decode(latents)
        
        # 扩散过程
        timesteps = torch.randint(
            0, self.scheduler.num_train_timesteps, (batch_size,),
            device=self.accelerator.device
        )
        
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        model_output = self.unet(
            noisy_latents,
            timesteps,
            panderm_features=panderm_features
        )['sample']
        
        # 计算损失
        loss_dict = self.criterion(
            model_output=model_output,
            noise_target=noise,
            recon_images=recon_images,
            target_images=images,
            vae_features=mu,
            panderm_features=panderm_features,
            timesteps=timesteps
        )
        
        # 转换为标量
        scalar_losses = {}
        for key, value in loss_dict.items():
            if torch.is_tensor(value):
                scalar_losses[key] = value.item()
            else:
                scalar_losses[key] = value
        
        return scalar_losses
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        
        checkpoint = {
            'epoch': epoch,
            'step': self.current_step,
            'unet_state_dict': self.accelerator.unwrap_model(self.unet).state_dict(),
            'vae_state_dict': self.accelerator.unwrap_model(self.vae).state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config.__dict__
        }
        
        if not self.config.model.panderm_freeze:
            checkpoint['panderm_state_dict'] = self.accelerator.unwrap_model(
                self.panderm_extractor
            ).state_dict()
        
        # 保存最新检查点
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            self.logger.info(f"保存最佳模型到 {best_path}")
        
        self.logger.info(f"保存检查点到 {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        
        checkpoint = torch.load(checkpoint_path, map_location=self.accelerator.device)
        
        self.accelerator.unwrap_model(self.unet).load_state_dict(checkpoint['unet_state_dict'])
        self.accelerator.unwrap_model(self.vae).load_state_dict(checkpoint['vae_state_dict'])
        
        if 'panderm_state_dict' in checkpoint and not self.config.model.panderm_freeze:
            self.accelerator.unwrap_model(self.panderm_extractor).load_state_dict(
                checkpoint['panderm_state_dict']
            )
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.current_step = checkpoint['step']
        self.best_loss = checkpoint['best_loss']
        
        self.logger.info(f"从检查点 {checkpoint_path} 恢复训练")
    
    def train(self):
        """主训练循环"""
        
        self.logger.info("开始训练...")
        
        for epoch in range(self.current_epoch, self.config.training.epochs):
            self.current_epoch = epoch
            
            # 训练一个epoch
            self.train_epoch(epoch)
            
            # 验证
            if (epoch + 1) % self.config.training.eval_every == 0:
                val_metrics, val_batch_for_vis = self.validate()
                
                # 记录验证指标
                if self.accelerator.is_main_process:
                    self.logger.info(f"Epoch {epoch+1} 验证结果:")
                    for key, value in val_metrics.items():
                        self.logger.info(f"  {key}: {value:.6f}")
                    
                    # 保存最佳模型
                    is_best = val_metrics['val_avg_loss'] < self.best_loss
                    if is_best:
                        self.best_loss = val_metrics['val_avg_loss']
                    
                    # 保存检查点
                    if (epoch + 1) % self.config.training.save_every == 0:
                        self.save_checkpoint(epoch + 1, is_best)
                    
                    # WandB可视化
                    if self.visualizer is not None and val_batch_for_vis is not None:
                        # 记录验证结果
                        self.visualizer.log_validation_results(
                            epoch + 1, val_batch_for_vis, val_metrics
                        )
                        
                        # 每5个epoch通过diffusion采样生成各类别图像
                        if (epoch + 1) % 5 == 0 and self.category_batches:
                            self.logger.info(f"第 {epoch+1} epoch: 开始diffusion采样生成各类别图像...")
                            try:
                                self.visualizer.log_generated_images_by_category(
                                    epoch + 1,
                                    self.category_batches,
                                    category_names=ISIC_CATEGORY_NAMES[:len(self.category_batches)],
                                    images_per_category=10,
                                    inference_steps=20  # 快速推理采样
                                )
                                self.logger.info(f"✓ 成功生成 {len(self.category_batches)} 个类别的diffusion采样图像")
                            except Exception as e:
                                self.logger.warning(f"diffusion采样生成失败: {e}")
                    
                    # Wandb日志
                    if wandb.run:
                        wandb.log({**val_metrics, 'epoch': epoch + 1})
            
            # 更新学习率
            self.lr_scheduler.step()
        
        self.logger.info("训练完成！")
    
    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        
        self.unet.train()
        self.vae.train()
        if not self.config.model.panderm_freeze:
            self.panderm_extractor.train()
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.config.training.epochs}",
            disable=not self.accelerator.is_local_main_process
        )
        
        epoch_loss = 0.0
        num_batches = 0
        
        train_batch_for_vis = None
        train_vis_data = None
        
        for batch_idx, batch in enumerate(progress_bar):
            self.current_step += 1
            
            # 训练步骤
            loss_dict, vis_data = self.train_step(batch)
            
            # 保存第一个批次用于可视化
            if batch_idx == 0 and self.visualizer is not None:
                train_batch_for_vis = batch
                train_vis_data = vis_data
            
            epoch_loss += loss_dict['total_loss']
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # 记录日志
            if self.current_step % self.config.training.log_every == 0:
                if self.accelerator.is_main_process:
                    log_dict = {
                        **loss_dict,
                        'epoch': epoch + 1,
                        'step': self.current_step,
                        'lr': self.optimizer.param_groups[0]['lr']
                    }
                    
                    # WandB可视化
                    if self.visualizer is not None:
                        # 记录损失曲线
                        self.visualizer.log_loss_curves(epoch + 1, loss_dict)
                        
                        # 记录学习率
                        self.visualizer.log_learning_rate(
                            epoch + 1, self.optimizer.param_groups[0]['lr']
                        )
                        
                        # 记录梯度统计
                        self.visualizer.log_model_gradients(epoch + 1, self.unet, "unet")
                        self.visualizer.log_model_gradients(epoch + 1, self.vae, "vae")
                        
                        # 记录特征分析
                        if train_vis_data is not None:
                            self.visualizer.log_feature_analysis(
                                epoch + 1,
                                train_vis_data['panderm_features'],
                                train_vis_data['vae_features'],
                                labels=train_batch_for_vis.get('target', None)
                            )
                    
                    if wandb.run:
                        wandb.log(log_dict)
        
        # Epoch结束可视化和日志
        if self.accelerator.is_main_process:
            avg_epoch_loss = epoch_loss / num_batches
            self.logger.info(f"Epoch {epoch+1} 平均损失: {avg_epoch_loss:.6f}")
            
            # 记录训练图像（每个epoch的第一个批次）
            if self.visualizer is not None and train_batch_for_vis is not None and train_vis_data is not None:
                self.visualizer.log_training_images(
                    epoch + 1,
                    train_batch_for_vis,
                    train_vis_data['recon_images']
                )


def main():
    """主函数"""
    # 加载配置
    config = Config()
    
    # 初始化训练器
    trainer = PanDermDiffusionTrainer(config)
    
    # 设置模型、数据、优化器
    trainer.setup_models()
    trainer.setup_data()
    trainer.setup_optimizer()
    trainer.setup_accelerator()
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()