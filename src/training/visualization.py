import torch
import torch.nn.functional as F
import wandb
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os
import torchvision.transforms as transforms
from .inference import PanDermDiffusionPipeline
from .diffusion_scheduler import DDIMScheduler


class WandBVisualizer:
    """WandB可视化工具"""
    
    def __init__(self, config, models: Dict = None):
        self.config = config
        self.models = models or {}
        
        # 图像变换
        self.denorm_transform = transforms.Compose([
            transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225]
            ),
            transforms.Lambda(lambda x: torch.clamp(x, 0, 1))
        ])
    
    def setup_pipeline(self, panderm_extractor, vae, unet):
        """设置推理管道用于生成"""
        scheduler = DDIMScheduler(
            num_train_timesteps=self.config.training.num_diffusion_steps,
            beta_schedule=self.config.training.noise_schedule
        )
        
        self.pipeline = PanDermDiffusionPipeline(
            panderm_extractor=panderm_extractor,
            vae=vae,
            unet=unet,
            scheduler=scheduler
        )
    
    def log_training_images(
        self,
        epoch: int,
        train_batch: Dict[str, torch.Tensor],
        recon_images: torch.Tensor,
        generated_samples: Optional[torch.Tensor] = None,
        max_images: int = 8
    ):
        """记录训练过程中的图像"""
        
        if not wandb.run:
            return
        
        original_images = train_batch['image'][:max_images]
        recon_images = recon_images[:max_images]
        
        # 反标准化图像
        original_vis = self._denormalize_images(original_images)
        recon_vis = self._denormalize_images(recon_images)
        
        # 创建对比图像
        comparison_images = []
        for i in range(len(original_vis)):
            # 横向拼接原图和重构图
            combined = torch.cat([original_vis[i], recon_vis[i]], dim=2)  # 沿width拼接
            comparison_images.append(combined)
        
        # 转换为wandb图像格式
        wandb_images = []
        for i, img in enumerate(comparison_images):
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            
            # 添加标签
            pil_img = Image.fromarray(img_np)
            draw = ImageDraw.Draw(pil_img)
            
            # 在图像上添加文本（如果字体可用）
            try:
                font = ImageFont.load_default()
                draw.text((10, 10), "Original", fill=(255, 255, 255), font=font)
                draw.text((img_np.shape[1]//2 + 10, 10), "Reconstructed", 
                         fill=(255, 255, 255), font=font)
            except:
                pass
            
            wandb_images.append(wandb.Image(
                pil_img, 
                caption=f"Sample {i+1} - Original vs Reconstructed"
            ))
        
        # 记录到wandb
        wandb.log({
            "train_reconstruction": wandb_images,
            "epoch": epoch
        })
        
        # 如果有生成的样本，也记录
        if generated_samples is not None:
            self._log_generated_samples(generated_samples[:max_images], epoch, "train_generated")
    
    def log_validation_results(
        self,
        epoch: int,
        val_batch: Dict[str, torch.Tensor],
        val_losses: Dict[str, float],
        max_images: int = 6
    ):
        """记录验证结果"""
        
        if not wandb.run:
            return
        
        # 记录验证损失
        val_log_dict = {f"val_{k}": v for k, v in val_losses.items()}
        val_log_dict["epoch"] = epoch
        wandb.log(val_log_dict)
        
        # 记录验证图像
        val_images = val_batch['image'][:max_images]
        val_images_vis = self._denormalize_images(val_images)
        
        wandb_images = []
        for i, img in enumerate(val_images_vis):
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            
            wandb_images.append(wandb.Image(
                Image.fromarray(img_np),
                caption=f"Validation Sample {i+1}"
            ))
        
        wandb.log({
            "validation_images": wandb_images,
            "epoch": epoch
        })
    
    def log_generated_images_by_category(
        self,
        epoch: int,
        reference_batches: Dict[str, torch.Tensor],
        category_names: Optional[List[str]] = None,
        images_per_category: int = 10,
        inference_steps: int = 20
    ):
        """
        通过diffusion采样按类别生成和记录图像
        
        这个方法会：
        1. 使用每个类别的参考图像提取PanDerm特征
        2. 用这些特征引导diffusion模型进行采样
        3. 生成全新的该类别皮肤病图像
        4. 在WandB中记录参考图像vs生成图像的对比
        
        注意：这里展示的是diffusion采样生成的新图像，不是原始数据集图像
        """
        
        if not wandb.run or not hasattr(self, 'pipeline'):
            return
        
        if category_names is None:
            category_names = [f"Category_{i}" for i in range(len(reference_batches))]
        
        all_generated_images = []
        category_logs = {}
        
        # 为每个类别通过diffusion采样生成全新图像
        for cat_idx, (category, ref_images) in enumerate(reference_batches.items()):
            cat_name = category_names[cat_idx] if cat_idx < len(category_names) else f"Category_{cat_idx}"
            
            try:
                # 通过diffusion pipeline采样生成图像
                # 过程：参考图像 → PanDerm特征提取 → 引导diffusion采样 → 生成新图像
                with torch.no_grad():
                    generated = self.pipeline.generate(
                        reference_images=ref_images,
                        num_samples=min(images_per_category, len(ref_images)),
                        num_inference_steps=inference_steps,
                        eta=0.0,
                        return_dict=True
                    )
                
                generated_images = generated['images']
                
                # 创建类别可视化
                category_vis = self._create_category_visualization(
                    ref_images[:images_per_category], 
                    generated_images,
                    cat_name
                )
                
                all_generated_images.extend(category_vis)
                
                # 单独记录每个类别的diffusion采样生成图像
                wandb_images = []
                for i, img in enumerate(generated_images):
                    img_np = img.permute(1, 2, 0).cpu().numpy()
                    img_np = (img_np * 255).astype(np.uint8)
                    
                    wandb_images.append(wandb.Image(
                        Image.fromarray(img_np),
                        caption=f"{cat_name} - Diffusion Generated {i+1}"
                    ))
                
                category_logs[f"generated_{cat_name.lower()}"] = wandb_images
                
            except Exception as e:
                print(f"生成 {cat_name} 类别图像时出错: {e}")
                continue
        
        # 记录所有类别的生成结果
        if all_generated_images:
            wandb.log({
                **category_logs,
                "generated_all_categories": all_generated_images,
                "epoch": epoch
            })
    
    def log_feature_analysis(
        self,
        epoch: int,
        panderm_features: torch.Tensor,
        vae_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ):
        """记录特征分析"""
        
        if not wandb.run:
            return
        
        # 计算特征统计
        panderm_mean = panderm_features.mean(dim=0)
        panderm_std = panderm_features.std(dim=0)
        vae_mean = vae_features.mean(dim=0)
        vae_std = vae_features.std(dim=0)
        
        # 特征相似性
        cosine_sim = F.cosine_similarity(
            panderm_features.mean(dim=0, keepdim=True),
            vae_features.mean(dim=0, keepdim=True)
        ).item()
        
        # 记录特征统计
        wandb.log({
            "features/panderm_mean_norm": torch.norm(panderm_mean).item(),
            "features/panderm_std_mean": panderm_std.mean().item(),
            "features/vae_mean_norm": torch.norm(vae_mean).item(),
            "features/vae_std_mean": vae_std.mean().item(),
            "features/cosine_similarity": cosine_sim,
            "epoch": epoch
        })
        
        # 如果特征维度不太大，创建特征分布图
        if panderm_features.shape[1] <= 100:
            self._log_feature_distributions(
                panderm_features, vae_features, epoch, labels
            )
    
    def log_loss_curves(
        self,
        epoch: int,
        train_losses: Dict[str, float],
        val_losses: Optional[Dict[str, float]] = None
    ):
        """记录损失曲线"""
        
        if not wandb.run:
            return
        
        log_dict = {}
        
        # 训练损失
        for key, value in train_losses.items():
            log_dict[f"train/{key}"] = value
        
        # 验证损失
        if val_losses:
            for key, value in val_losses.items():
                log_dict[f"val/{key}"] = value
        
        log_dict["epoch"] = epoch
        wandb.log(log_dict)
    
    def log_learning_rate(self, epoch: int, lr: float):
        """记录学习率"""
        if wandb.run:
            wandb.log({"learning_rate": lr, "epoch": epoch})
    
    def log_model_gradients(self, epoch: int, model: torch.nn.Module, model_name: str):
        """记录模型梯度统计"""
        
        if not wandb.run:
            return
        
        total_norm = 0.0
        param_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        total_norm = total_norm ** (1. / 2)
        
        wandb.log({
            f"gradients/{model_name}_total_norm": total_norm,
            f"gradients/{model_name}_param_count": param_count,
            "epoch": epoch
        })
    
    def _denormalize_images(self, images: torch.Tensor) -> torch.Tensor:
        """反标准化图像到[0,1]范围"""
        if images.min() < 0:  # 如果图像被标准化了
            return self.denorm_transform(images)
        else:
            return torch.clamp(images, 0, 1)
    
    def _log_generated_samples(
        self, 
        samples: torch.Tensor, 
        epoch: int, 
        prefix: str
    ):
        """记录生成样本"""
        
        samples_vis = self._denormalize_images(samples)
        
        wandb_images = []
        for i, img in enumerate(samples_vis):
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            
            wandb_images.append(wandb.Image(
                Image.fromarray(img_np),
                caption=f"Generated Sample {i+1}"
            ))
        
        wandb.log({
            prefix: wandb_images,
            "epoch": epoch
        })
    
    def _create_category_visualization(
        self,
        reference_images: torch.Tensor,
        generated_images: torch.Tensor,
        category_name: str
    ) -> List[wandb.Image]:
        """创建类别对比可视化"""
        
        ref_vis = self._denormalize_images(reference_images)
        gen_vis = self._denormalize_images(generated_images)
        
        wandb_images = []
        
        # 创建网格对比图
        max_pairs = min(len(ref_vis), len(gen_vis))
        
        for i in range(max_pairs):
            # 水平拼接参考图像和生成图像
            ref_img = ref_vis[i].permute(1, 2, 0).cpu().numpy()
            gen_img = gen_vis[i].permute(1, 2, 0).cpu().numpy()
            
            ref_img = (ref_img * 255).astype(np.uint8)
            gen_img = (gen_img * 255).astype(np.uint8)
            
            # 创建拼接图像
            combined = np.concatenate([ref_img, gen_img], axis=1)
            pil_img = Image.fromarray(combined)
            
            # 添加标签
            draw = ImageDraw.Draw(pil_img)
            try:
                font = ImageFont.load_default()
                draw.text((10, 10), "Reference", fill=(255, 255, 255), font=font)
                draw.text((ref_img.shape[1] + 10, 10), "Generated", 
                         fill=(255, 255, 255), font=font)
            except:
                pass
            
            wandb_images.append(wandb.Image(
                pil_img,
                caption=f"{category_name} - Pair {i+1}"
            ))
        
        return wandb_images
    
    def _log_feature_distributions(
        self,
        panderm_features: torch.Tensor,
        vae_features: torch.Tensor,
        epoch: int,
        labels: Optional[torch.Tensor] = None
    ):
        """记录特征分布图"""
        
        # 使用PCA降维到2D
        try:
            from sklearn.decomposition import PCA
            
            # 合并特征进行PCA
            all_features = torch.cat([panderm_features, vae_features], dim=0)
            all_features_np = all_features.cpu().numpy()
            
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(all_features_np)
            
            n_panderm = len(panderm_features)
            panderm_2d = features_2d[:n_panderm]
            vae_2d = features_2d[n_panderm:]
            
            # 创建散点图
            plt.figure(figsize=(10, 8))
            plt.scatter(panderm_2d[:, 0], panderm_2d[:, 1], 
                       alpha=0.6, label='PanDerm Features', s=20)
            plt.scatter(vae_2d[:, 0], vae_2d[:, 1], 
                       alpha=0.6, label='VAE Features', s=20)
            
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.title('Feature Space Visualization (PCA)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 记录到wandb
            wandb.log({
                "features/pca_plot": wandb.Image(plt),
                "epoch": epoch
            })
            
            plt.close()
            
        except ImportError:
            print("sklearn not available, skipping feature visualization")
        except Exception as e:
            print(f"Feature visualization error: {e}")


def create_category_batches(
    dataloader,
    num_categories: int = 5,
    samples_per_category: int = 10,
    device: torch.device = None
) -> Dict[str, torch.Tensor]:
    """从数据加载器创建类别批次"""
    
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 如果数据集有标签信息，按标签分类
    category_batches = {}
    category_counts = defaultdict(list)
    
    # 简单的按批次索引分类（如果没有真实标签）
    batch_idx = 0
    for batch in dataloader:
        images = batch['image'].to(device)
        
        # 如果有目标标签，使用标签分类
        if 'target' in batch and batch['target'].max() > 0:
            targets = batch['target']
            for i, target in enumerate(targets):
                cat_key = f"category_{target.item()}"
                if len(category_counts[cat_key]) < samples_per_category:
                    category_counts[cat_key].append(images[i])
        else:
            # 否则按批次轮询分配
            cat_key = f"category_{batch_idx % num_categories}"
            for i, img in enumerate(images):
                if len(category_counts[cat_key]) < samples_per_category:
                    category_counts[cat_key].append(img)
                
                # 轮换到下一个类别
                batch_idx = (batch_idx + 1) % num_categories
                cat_key = f"category_{batch_idx % num_categories}"
        
        # 检查是否收集够了所有类别
        if all(len(imgs) >= samples_per_category 
               for imgs in category_counts.values()) and \
           len(category_counts) >= num_categories:
            break
    
    # 转换为张量
    for cat_key, img_list in category_counts.items():
        if img_list:
            category_batches[cat_key] = torch.stack(img_list[:samples_per_category])
    
    return category_batches


# ISIC数据集的皮肤病类别名称
ISIC_CATEGORY_NAMES = [
    "Melanoma",           # 黑色素瘤
    "Melanocytic_nevus",  # 色素痣
    "Basal_cell_carcinoma", # 基底细胞癌
    "Actinic_keratosis",  # 光化性角化病
    "Benign_keratosis",   # 良性角化病
    "Dermatofibroma",     # 皮肤纤维瘤
    "Vascular_lesion",    # 血管病变
    "Squamous_cell_carcinoma", # 鳞状细胞癌
    "Other"               # 其他
]


if __name__ == "__main__":
    # 测试可视化工具
    from configs.config import Config
    
    config = Config()
    visualizer = WandBVisualizer(config)
    
    # 创建测试数据
    batch_size = 4
    test_batch = {
        'image': torch.randn(batch_size, 3, 224, 224),
        'target': torch.randint(0, 5, (batch_size,))
    }
    
    recon_images = torch.randn(batch_size, 3, 224, 224)
    
    print("可视化工具测试:")
    print("✓ WandBVisualizer初始化成功")
    
    # 测试图像反标准化
    denorm_images = visualizer._denormalize_images(test_batch['image'])
    print(f"✓ 图像反标准化测试: {denorm_images.shape}, 范围: [{denorm_images.min():.3f}, {denorm_images.max():.3f}]")
    
    # 测试类别批次创建
    device = torch.device("cpu")
    
    # 模拟数据加载器
    class MockDataLoader:
        def __init__(self):
            self.data = []
            for i in range(3):  # 3个批次
                batch = {
                    'image': torch.randn(8, 3, 224, 224),
                    'target': torch.randint(0, 3, (8,))
                }
                self.data.append(batch)
        
        def __iter__(self):
            return iter(self.data)
    
    mock_loader = MockDataLoader()
    category_batches = create_category_batches(
        mock_loader, 
        num_categories=3, 
        samples_per_category=5,
        device=device
    )
    
    print(f"✓ 类别批次创建测试: {len(category_batches)} 个类别")
    for cat, images in category_batches.items():
        print(f"  {cat}: {images.shape}")
    
    print("\n可视化工具测试完成！")