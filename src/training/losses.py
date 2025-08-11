import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import torchvision.transforms as transforms
from torchvision.models import vgg19


class DiffusionLoss(nn.Module):
    """Diffusion模型损失函数"""
    
    def __init__(
        self,
        prediction_type: str = "epsilon",
        loss_type: str = "l2",  # "l1", "l2", "huber"
        huber_delta: float = 0.1
    ):
        super().__init__()
        self.prediction_type = prediction_type
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        
    def forward(
        self,
        model_output: torch.Tensor,
        target: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        sample_weight: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算扩散损失
        
        Args:
            model_output: 模型预测输出
            target: 真实目标（噪声或原始样本）
            timesteps: 时间步（可用于加权）
            sample_weight: 样本权重
        """
        
        if self.loss_type == "l1":
            loss = F.l1_loss(model_output, target, reduction='none')
        elif self.loss_type == "l2":
            loss = F.mse_loss(model_output, target, reduction='none')
        elif self.loss_type == "huber":
            loss = F.huber_loss(model_output, target, delta=self.huber_delta, reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # 时间步加权（可选）
        if timesteps is not None:
            # 早期时间步（高噪声）权重更高
            time_weights = 1.0 + 0.5 * (timesteps.float() / 1000.0)
            time_weights = time_weights.view(-1, 1, 1, 1)
            loss = loss * time_weights
        
        # 样本权重（可选）
        if sample_weight is not None:
            loss = loss * sample_weight.view(-1, 1, 1, 1)
        
        # 计算平均损失
        avg_loss = loss.mean()
        
        return {
            'diffusion_loss': avg_loss,
            'loss_map': loss
        }


class REPALoss(nn.Module):
    """表示对齐损失（REPA Loss）"""
    
    def __init__(
        self,
        alignment_type: str = "cosine",  # "cosine", "l2", "kl"
        temperature: float = 0.07,
        normalize_features: bool = True
    ):
        super().__init__()
        self.alignment_type = alignment_type
        self.temperature = temperature
        self.normalize_features = normalize_features
        
    def forward(
        self,
        vae_features: torch.Tensor,
        panderm_features: torch.Tensor,
        feature_masks: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算表示对齐损失
        
        Args:
            vae_features: VAE编码器特征 [B, D] 或 [B, H*W, D]
            panderm_features: PanDerm特征 [B, D] 或 [B, N, D]  
            feature_masks: 特征掩码（可选）
        """
        
        # 特征标准化
        if self.normalize_features:
            if vae_features.dim() == 3:  # 空间特征
                vae_features = F.normalize(vae_features, p=2, dim=-1)
            else:  # 全局特征
                vae_features = F.normalize(vae_features, p=2, dim=-1)
                
            if panderm_features.dim() == 3:
                panderm_features = F.normalize(panderm_features, p=2, dim=-1)
            else:
                panderm_features = F.normalize(panderm_features, p=2, dim=-1)
        
        if self.alignment_type == "cosine":
            loss = self._cosine_alignment_loss(vae_features, panderm_features, feature_masks)
        elif self.alignment_type == "l2":
            loss = self._l2_alignment_loss(vae_features, panderm_features, feature_masks)
        elif self.alignment_type == "kl":
            loss = self._kl_alignment_loss(vae_features, panderm_features, feature_masks)
        else:
            raise ValueError(f"Unknown alignment type: {self.alignment_type}")
        
        return {
            'repa_loss': loss,
            'alignment_type': self.alignment_type
        }
    
    def _cosine_alignment_loss(
        self, 
        vae_feat: torch.Tensor, 
        panderm_feat: torch.Tensor,
        masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """余弦相似度对齐损失"""
        
        if vae_feat.dim() == 3 and panderm_feat.dim() == 2:
            # 空间特征 vs 全局特征：使用全局平均池化
            panderm_feat = panderm_feat.unsqueeze(1)  # [B, 1, D]
            similarities = torch.cosine_similarity(vae_feat, panderm_feat, dim=-1)  # [B, H*W]
            
            if masks is not None:
                similarities = similarities * masks
                loss = 1.0 - similarities.sum() / masks.sum()
            else:
                loss = 1.0 - similarities.mean()
                
        elif vae_feat.dim() == 2 and panderm_feat.dim() == 2:
            # 全局特征 vs 全局特征
            similarities = torch.cosine_similarity(vae_feat, panderm_feat, dim=-1)  # [B,]
            loss = 1.0 - similarities.mean()
            
        else:
            # 空间特征 vs 空间特征：使用注意力权重
            B, N1, D = vae_feat.shape
            N2 = panderm_feat.shape[1]
            
            # 计算注意力矩阵
            attention = torch.matmul(vae_feat, panderm_feat.transpose(-2, -1)) / (D ** 0.5)
            attention = F.softmax(attention, dim=-1)  # [B, N1, N2]
            
            # 加权特征对齐
            aligned_panderm = torch.matmul(attention, panderm_feat)  # [B, N1, D]
            similarities = torch.cosine_similarity(vae_feat, aligned_panderm, dim=-1)  # [B, N1]
            
            if masks is not None:
                similarities = similarities * masks
                loss = 1.0 - similarities.sum() / masks.sum()
            else:
                loss = 1.0 - similarities.mean()
        
        return loss
    
    def _l2_alignment_loss(
        self, 
        vae_feat: torch.Tensor, 
        panderm_feat: torch.Tensor,
        masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """L2距离对齐损失"""
        
        if vae_feat.dim() == 3 and panderm_feat.dim() == 2:
            # 广播计算L2距离
            panderm_feat = panderm_feat.unsqueeze(1)  # [B, 1, D]
            distances = F.mse_loss(vae_feat, panderm_feat, reduction='none').mean(dim=-1)  # [B, H*W]
            
            if masks is not None:
                distances = distances * masks
                loss = distances.sum() / masks.sum()
            else:
                loss = distances.mean()
                
        elif vae_feat.dim() == 2 and panderm_feat.dim() == 2:
            loss = F.mse_loss(vae_feat, panderm_feat)
        else:
            # 使用注意力对齐后计算L2损失
            B, N1, D = vae_feat.shape
            N2 = panderm_feat.shape[1]
            
            attention = torch.matmul(vae_feat, panderm_feat.transpose(-2, -1)) / (D ** 0.5)
            attention = F.softmax(attention, dim=-1)
            
            aligned_panderm = torch.matmul(attention, panderm_feat)
            distances = F.mse_loss(vae_feat, aligned_panderm, reduction='none').mean(dim=-1)
            
            if masks is not None:
                distances = distances * masks
                loss = distances.sum() / masks.sum()
            else:
                loss = distances.mean()
        
        return loss
    
    def _kl_alignment_loss(
        self, 
        vae_feat: torch.Tensor, 
        panderm_feat: torch.Tensor,
        masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """KL散度对齐损失"""
        
        # 将特征转换为概率分布
        vae_prob = F.softmax(vae_feat / self.temperature, dim=-1)
        panderm_prob = F.softmax(panderm_feat / self.temperature, dim=-1)
        
        if vae_feat.dim() == 3 and panderm_feat.dim() == 2:
            panderm_prob = panderm_prob.unsqueeze(1).expand_as(vae_prob)
        
        # 计算KL散度
        kl_div = F.kl_div(
            torch.log(vae_prob + 1e-8), 
            panderm_prob, 
            reduction='none'
        ).sum(dim=-1)
        
        if masks is not None:
            kl_div = kl_div * masks
            loss = kl_div.sum() / masks.sum()
        else:
            loss = kl_div.mean()
        
        return loss


class PerceptualLoss(nn.Module):
    """感知损失（基于预训练VGG网络）"""
    
    def __init__(
        self,
        layers: Tuple[str, ...] = ('conv_1', 'conv_2', 'conv_3', 'conv_4'),
        weights: Optional[Tuple[float, ...]] = None,
        normalize_input: bool = True
    ):
        super().__init__()
        
        self.layers = layers
        self.weights = weights or [1.0] * len(layers)
        self.normalize_input = normalize_input
        
        # 加载预训练VGG19
        vgg = vgg19(pretrained=True).features
        self.vgg_layers = nn.ModuleDict()
        
        layer_mapping = {
            'conv_1': '4',   # relu1_2
            'conv_2': '9',   # relu2_2
            'conv_3': '18',  # relu3_4
            'conv_4': '27',  # relu4_4
            'conv_5': '36',  # relu5_4
        }
        
        for layer_name in layers:
            if layer_name in layer_mapping:
                layer_idx = int(layer_mapping[layer_name])
                self.vgg_layers[layer_name] = nn.Sequential(*list(vgg.children())[:layer_idx+1])
        
        # 冻结VGG参数
        for param in self.vgg_layers.parameters():
            param.requires_grad = False
        
        # ImageNet标准化
        if normalize_input:
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """输入标准化"""
        if self.normalize_input:
            # 假设输入在[-1, 1]范围内，转换到[0, 1]
            x = (x + 1.0) / 2.0
            x = (x - self.mean) / self.std
        return x
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        计算感知损失
        
        Args:
            pred: 预测图像 [B, C, H, W]
            target: 目标图像 [B, C, H, W]
        """
        
        pred_norm = self._normalize_input(pred)
        target_norm = self._normalize_input(target)
        
        total_loss = 0.0
        layer_losses = {}
        
        for i, layer_name in enumerate(self.layers):
            # 提取特征
            pred_feat = self.vgg_layers[layer_name](pred_norm)
            target_feat = self.vgg_layers[layer_name](target_norm)
            
            # 计算L1损失
            layer_loss = F.l1_loss(pred_feat, target_feat)
            layer_losses[f'{layer_name}_loss'] = layer_loss
            
            total_loss += self.weights[i] * layer_loss
        
        return {
            'perceptual_loss': total_loss,
            **layer_losses
        }


class CombinedLoss(nn.Module):
    """组合损失函数"""
    
    def __init__(
        self,
        alpha_diffusion: float = 1.0,
        beta_recon: float = 0.5,
        gamma_repa: float = 0.3,
        delta_perceptual: float = 0.2,
        use_perceptual: bool = True,
        use_repa: bool = True
    ):
        super().__init__()
        
        self.alpha_diffusion = alpha_diffusion
        self.beta_recon = beta_recon
        self.gamma_repa = gamma_repa
        self.delta_perceptual = delta_perceptual
        self.use_perceptual = use_perceptual
        self.use_repa = use_repa
        
        # 损失模块
        self.diffusion_loss = DiffusionLoss()
        
        if use_repa:
            self.repa_loss = REPALoss()
        
        if use_perceptual:
            self.perceptual_loss = PerceptualLoss()
    
    def forward(
        self,
        model_output: torch.Tensor,
        noise_target: torch.Tensor,
        recon_images: Optional[torch.Tensor] = None,
        target_images: Optional[torch.Tensor] = None,
        vae_features: Optional[torch.Tensor] = None,
        panderm_features: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算组合损失
        
        Args:
            model_output: UNet预测的噪声
            noise_target: 真实噪声
            recon_images: VAE重构图像
            target_images: 目标图像
            vae_features: VAE特征
            panderm_features: PanDerm特征
            timesteps: 时间步
        """
        
        losses = {}
        total_loss = 0.0
        
        # 1. Diffusion损失
        diff_loss_dict = self.diffusion_loss(model_output, noise_target, timesteps)
        losses.update(diff_loss_dict)
        total_loss += self.alpha_diffusion * diff_loss_dict['diffusion_loss']
        
        # 2. 重构损失
        if recon_images is not None and target_images is not None:
            recon_loss = F.l1_loss(recon_images, target_images) + F.mse_loss(recon_images, target_images)
            losses['recon_loss'] = recon_loss
            total_loss += self.beta_recon * recon_loss
        
        # 3. REPA损失
        if self.use_repa and vae_features is not None and panderm_features is not None:
            repa_loss_dict = self.repa_loss(vae_features, panderm_features)
            losses.update(repa_loss_dict)
            total_loss += self.gamma_repa * repa_loss_dict['repa_loss']
        
        # 4. 感知损失
        if self.use_perceptual and recon_images is not None and target_images is not None:
            perceptual_loss_dict = self.perceptual_loss(recon_images, target_images)
            losses.update(perceptual_loss_dict)
            total_loss += self.delta_perceptual * perceptual_loss_dict['perceptual_loss']
        
        losses['total_loss'] = total_loss
        return losses


if __name__ == "__main__":
    # 测试损失函数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = 2
    channels = 4
    height, width = 64, 64
    
    # 测试数据
    model_output = torch.randn(batch_size, channels, height, width).to(device)
    noise_target = torch.randn(batch_size, channels, height, width).to(device)
    recon_images = torch.randn(batch_size, 3, 256, 256).to(device)
    target_images = torch.randn(batch_size, 3, 256, 256).to(device)
    vae_features = torch.randn(batch_size, 768).to(device)
    panderm_features = torch.randn(batch_size, 768).to(device)
    timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
    
    # 测试组合损失
    combined_loss = CombinedLoss().to(device)
    
    loss_dict = combined_loss(
        model_output=model_output,
        noise_target=noise_target,
        recon_images=recon_images,
        target_images=target_images,
        vae_features=vae_features,
        panderm_features=panderm_features,
        timesteps=timesteps
    )
    
    print("损失函数测试:")
    for key, value in loss_dict.items():
        if torch.is_tensor(value):
            print(f"  {key}: {value.item():.6f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n损失函数测试完成！")