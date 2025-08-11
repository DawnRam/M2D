import torch
import torch.nn.functional as F
from typing import Dict, Optional, List, Union, Tuple
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

from ..models import PanDermFeatureExtractor, VAE, UNet2D
from .diffusion_scheduler import DDPMScheduler, DDIMScheduler
from configs.config import Config


class PanDermDiffusionPipeline:
    """PanDerm-Guided Diffusion推理管道"""
    
    def __init__(
        self,
        panderm_extractor: PanDermFeatureExtractor,
        vae: VAE,
        unet: UNet2D,
        scheduler: Union[DDPMScheduler, DDIMScheduler],
        device: torch.device = None
    ):
        self.panderm_extractor = panderm_extractor
        self.vae = vae
        self.unet = unet
        self.scheduler = scheduler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 移动模型到设备
        self.panderm_extractor.to(self.device)
        self.vae.to(self.device)
        self.unet.to(self.device)
        
        # 设置为评估模式
        self.panderm_extractor.eval()
        self.vae.eval()
        self.unet.eval()
    
    @torch.no_grad()
    def generate(
        self,
        reference_images: Optional[torch.Tensor] = None,
        panderm_features: Optional[torch.Tensor] = None,
        num_samples: int = 4,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        生成新的皮肤镜图像
        
        Args:
            reference_images: 参考图像 [B, 3, H, W]，用于提取PanDerm特征
            panderm_features: 预计算的PanDerm特征 [B, D]
            num_samples: 生成样本数量
            num_inference_steps: 推理步数
            guidance_scale: 引导强度
            eta: DDIM eta参数
            generator: 随机数生成器
            return_dict: 是否返回字典格式
        """
        
        # 1. 获取PanDerm特征
        if panderm_features is None:
            if reference_images is None:
                raise ValueError("必须提供reference_images或panderm_features之一")
            
            # 提取PanDerm特征
            panderm_features = self.panderm_extractor(reference_images)['global']  # [B, D]
            batch_size = reference_images.shape[0]
        else:
            batch_size = panderm_features.shape[0]
        
        # 如果需要生成更多样本，复制特征
        if num_samples > batch_size:
            repeat_factor = num_samples // batch_size
            remainder = num_samples % batch_size
            
            panderm_features_list = [panderm_features.repeat(repeat_factor, 1)]
            if remainder > 0:
                panderm_features_list.append(panderm_features[:remainder])
            
            panderm_features = torch.cat(panderm_features_list, dim=0)
            batch_size = num_samples
        elif num_samples < batch_size:
            panderm_features = panderm_features[:num_samples]
            batch_size = num_samples
        
        # 2. 设置调度器
        if isinstance(self.scheduler, DDIMScheduler):
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps = self.scheduler.timesteps
        else:
            timesteps = torch.linspace(
                self.scheduler.num_train_timesteps - 1, 0, num_inference_steps, dtype=torch.int64
            )
        
        # 3. 初始化潜在变量
        # 假设VAE的潜在空间大小为64x64
        latent_shape = (batch_size, 4, 64, 64)  # [B, C, H, W]
        latents = torch.randn(
            latent_shape,
            generator=generator,
            device=self.device,
            dtype=self.unet.dtype
        )
        
        # 缩放初始噪声
        if isinstance(self.scheduler, DDPMScheduler):
            latents = latents * self.scheduler.init_noise_sigma
        
        # 4. 去噪循环
        progress_bar = tqdm(timesteps, desc="生成中")
        
        for i, t in enumerate(progress_bar):
            timestep = t.unsqueeze(0).repeat(batch_size).to(self.device)
            
            # UNet预测
            model_output = self.unet(
                latents,
                timestep,
                panderm_features=panderm_features
            )['sample']
            
            # 调度器步骤
            scheduler_output = self.scheduler.step(
                model_output=model_output,
                timestep=int(t),
                sample=latents,
                eta=eta,
                generator=generator
            )
            
            latents = scheduler_output['prev_sample']
        
        # 5. VAE解码
        images = self.vae.decode(latents)
        
        # 6. 后处理
        images = self._postprocess_images(images)
        
        if return_dict:
            return {
                'images': images,
                'latents': latents,
                'panderm_features': panderm_features
            }
        else:
            return images
    
    @torch.no_grad()
    def image_to_image(
        self,
        init_images: torch.Tensor,
        reference_images: Optional[torch.Tensor] = None,
        panderm_features: Optional[torch.Tensor] = None,
        strength: float = 0.75,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        图像到图像生成（基于初始图像）
        
        Args:
            init_images: 初始图像 [B, 3, H, W]
            reference_images: 参考图像（用于特征提取）
            panderm_features: 预计算的特征
            strength: 变形强度 (0-1)
            num_inference_steps: 推理步数
            guidance_scale: 引导强度
            eta: DDIM eta参数
            generator: 随机数生成器
            return_dict: 是否返回字典
        """
        
        batch_size = init_images.shape[0]
        
        # 1. 获取PanDerm特征
        if panderm_features is None:
            if reference_images is None:
                # 使用初始图像作为参考
                reference_images = init_images
            
            panderm_features = self.panderm_extractor(reference_images)['global']
        
        # 2. VAE编码初始图像
        mu, logvar = self.vae.encode(init_images)
        init_latents = self.vae.reparameterize(mu, logvar)
        
        # 3. 设置调度器和时间步
        if isinstance(self.scheduler, DDIMScheduler):
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps = self.scheduler.timesteps
        else:
            timesteps = torch.linspace(
                self.scheduler.num_train_timesteps - 1, 0, num_inference_steps, dtype=torch.int64
            )
        
        # 计算开始时间步（基于strength）
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = timesteps[t_start:]
        
        # 4. 向初始潜在变量添加噪声
        noise = torch.randn_like(init_latents, generator=generator)
        init_latents = self.scheduler.add_noise(
            init_latents, noise, timesteps[0:1].repeat(batch_size)
        )
        
        latents = init_latents
        
        # 5. 去噪循环
        progress_bar = tqdm(timesteps, desc="图像转换中")
        
        for i, t in enumerate(progress_bar):
            timestep = t.unsqueeze(0).repeat(batch_size).to(self.device)
            
            # UNet预测
            model_output = self.unet(
                latents,
                timestep,
                panderm_features=panderm_features
            )['sample']
            
            # 调度器步骤
            scheduler_output = self.scheduler.step(
                model_output=model_output,
                timestep=int(t),
                sample=latents,
                eta=eta,
                generator=generator
            )
            
            latents = scheduler_output['prev_sample']
        
        # 6. VAE解码
        images = self.vae.decode(latents)
        
        # 7. 后处理
        images = self._postprocess_images(images)
        
        if return_dict:
            return {
                'images': images,
                'latents': latents,
                'panderm_features': panderm_features,
                'init_images': init_images
            }
        else:
            return images
    
    @torch.no_grad()
    def interpolate(
        self,
        reference_images_a: torch.Tensor,
        reference_images_b: torch.Tensor,
        num_interpolations: int = 8,
        num_inference_steps: int = 50,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        特征插值生成
        
        Args:
            reference_images_a: 第一组参考图像
            reference_images_b: 第二组参考图像
            num_interpolations: 插值点数量
            num_inference_steps: 推理步数
            eta: DDIM eta参数
            generator: 随机数生成器
        """
        
        # 提取特征
        features_a = self.panderm_extractor(reference_images_a)['global']
        features_b = self.panderm_extractor(reference_images_b)['global']
        
        # 线性插值
        interpolation_ratios = torch.linspace(0, 1, num_interpolations)
        interpolated_images = []
        
        for ratio in interpolation_ratios:
            interpolated_features = (1 - ratio) * features_a + ratio * features_b
            
            # 生成图像
            generated = self.generate(
                panderm_features=interpolated_features,
                num_samples=1,
                num_inference_steps=num_inference_steps,
                eta=eta,
                generator=generator,
                return_dict=False
            )
            
            interpolated_images.append(generated)
        
        return torch.cat(interpolated_images, dim=0)
    
    def _postprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """后处理图像"""
        # 裁剪到[-1, 1]范围
        images = torch.clamp(images, -1.0, 1.0)
        
        # 转换到[0, 1]范围
        images = (images + 1.0) / 2.0
        
        return images
    
    def save_images(
        self,
        images: torch.Tensor,
        output_dir: str,
        prefix: str = "generated",
        format: str = "png"
    ):
        """保存图像到文件"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 转换为numpy数组
        images_np = images.cpu().numpy().transpose(0, 2, 3, 1)  # [B, H, W, C]
        images_np = (images_np * 255).astype(np.uint8)
        
        for i, img_np in enumerate(images_np):
            img_pil = Image.fromarray(img_np)
            filename = f"{prefix}_{i:04d}.{format}"
            filepath = os.path.join(output_dir, filename)
            img_pil.save(filepath)
            
        print(f"保存了 {len(images_np)} 张图像到 {output_dir}")


def load_pipeline_from_checkpoint(
    checkpoint_path: str,
    config: Config,
    scheduler_type: str = "ddim",  # "ddpm" or "ddim"
    device: torch.device = None
) -> PanDermDiffusionPipeline:
    """从检查点加载推理管道"""
    
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 初始化模型
    panderm_extractor = PanDermFeatureExtractor(
        model_name=config.model.panderm_model,
        freeze_backbone=True,  # 推理时总是冻结
        feature_dim=config.model.feature_dim
    )
    
    vae = VAE(
        in_channels=3,
        latent_channels=config.model.vae_channels,
        base_channels=128
    )
    
    unet = UNet2D(
        in_channels=config.model.vae_channels,
        out_channels=config.model.vae_channels,
        model_channels=config.model.unet_channels,
        time_embed_dim=config.model.time_embed_dim,
        context_dim=config.model.feature_dim,
        use_feature_fusion=True,
        fusion_type=config.model.fusion_type
    )
    
    # 加载权重
    vae.load_state_dict(checkpoint['vae_state_dict'])
    unet.load_state_dict(checkpoint['unet_state_dict'])
    
    if 'panderm_state_dict' in checkpoint:
        panderm_extractor.load_state_dict(checkpoint['panderm_state_dict'])
    
    # 初始化调度器
    if scheduler_type == "ddim":
        scheduler = DDIMScheduler(
            num_train_timesteps=config.training.num_diffusion_steps,
            beta_schedule=config.training.noise_schedule
        )
    else:
        scheduler = DDPMScheduler(
            num_train_timesteps=config.training.num_diffusion_steps,
            beta_schedule=config.training.noise_schedule
        )
    
    # 创建管道
    pipeline = PanDermDiffusionPipeline(
        panderm_extractor=panderm_extractor,
        vae=vae,
        unet=unet,
        scheduler=scheduler,
        device=device
    )
    
    return pipeline


if __name__ == "__main__":
    # 测试推理管道
    from configs.config import Config
    
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型（这里使用随机权重进行测试）
    panderm_extractor = PanDermFeatureExtractor(
        model_name="panderm-large",
        freeze_backbone=True,
        feature_dim=768
    )
    
    vae = VAE(in_channels=3, latent_channels=4, base_channels=64)
    
    unet = UNet2D(
        in_channels=4,
        out_channels=4,
        model_channels=64,
        context_dim=768,
        use_feature_fusion=True
    )
    
    scheduler = DDIMScheduler(num_train_timesteps=1000)
    
    # 创建管道
    pipeline = PanDermDiffusionPipeline(
        panderm_extractor=panderm_extractor,
        vae=vae,
        unet=unet,
        scheduler=scheduler,
        device=device
    )
    
    # 测试生成
    print("测试图像生成...")
    reference_images = torch.randn(2, 3, 224, 224).to(device)
    
    with torch.no_grad():
        generated = pipeline.generate(
            reference_images=reference_images,
            num_samples=4,
            num_inference_steps=20  # 减少步数以便快速测试
        )
        
        print(f"生成图像形状: {generated['images'].shape}")
        print(f"PanDerm特征形状: {generated['panderm_features'].shape}")
    
    # 测试图像到图像
    print("\n测试图像到图像...")
    init_images = torch.randn(1, 3, 256, 256).to(device)
    
    with torch.no_grad():
        img2img_result = pipeline.image_to_image(
            init_images=init_images,
            strength=0.75,
            num_inference_steps=20
        )
        
        print(f"图像转换结果形状: {img2img_result['images'].shape}")
    
    print("\n推理管道测试完成！")