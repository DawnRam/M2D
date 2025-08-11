import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """自注意力块"""
    
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        h = self.norm(x)
        qkv = self.qkv(h)
        
        # Reshape for multi-head attention
        qkv = qkv.view(B, 3, self.num_heads, self.head_dim, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # [3, B, num_heads, H*W, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)  # [B, num_heads, H*W, head_dim]
        out = out.transpose(-2, -1).contiguous().view(B, C, H, W)
        
        out = self.proj(out)
        return x + out


class VAEEncoder(nn.Module):
    """VAE编码器"""
    
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 128,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16,),
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.base_channels = base_channels
        
        # 初始卷积
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # 下采样层
        self.down_blocks = nn.ModuleList()
        in_ch = base_channels
        
        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            
            # 残差块
            res_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                res_blocks.append(ResidualBlock(in_ch, out_ch, dropout))
                in_ch = out_ch
            
            # 注意力块
            if 2**i in attention_resolutions:
                res_blocks.append(AttentionBlock(out_ch))
            
            self.down_blocks.append(res_blocks)
            
            # 下采样（除了最后一层）
            if i < len(channel_multipliers) - 1:
                self.down_blocks.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))
        
        # 中间层
        mid_channels = base_channels * channel_multipliers[-1]
        self.mid_block1 = ResidualBlock(mid_channels, mid_channels, dropout)
        self.mid_attn = AttentionBlock(mid_channels)
        self.mid_block2 = ResidualBlock(mid_channels, mid_channels, dropout)
        
        # 输出层（均值和方差）
        self.norm_out = nn.GroupNorm(32, mid_channels)
        self.conv_out = nn.Conv2d(mid_channels, 2 * latent_channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入图像 [B, C, H, W]
            
        Returns:
            mu: 均值 [B, latent_channels, H', W']
            logvar: 对数方差 [B, latent_channels, H', W']
        """
        h = self.conv_in(x)
        
        # 下采样
        for block in self.down_blocks:
            if isinstance(block, nn.ModuleList):
                for layer in block:
                    h = layer(h)
            else:
                h = block(h)
        
        # 中间层
        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)
        
        # 输出
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        mu, logvar = torch.chunk(h, 2, dim=1)
        
        return mu, logvar


class VAEDecoder(nn.Module):
    """VAE解码器"""
    
    def __init__(
        self,
        out_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 128,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16,),
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.base_channels = base_channels
        
        # 输入层
        mid_channels = base_channels * channel_multipliers[-1]
        self.conv_in = nn.Conv2d(latent_channels, mid_channels, 3, padding=1)
        
        # 中间层
        self.mid_block1 = ResidualBlock(mid_channels, mid_channels, dropout)
        self.mid_attn = AttentionBlock(mid_channels)
        self.mid_block2 = ResidualBlock(mid_channels, mid_channels, dropout)
        
        # 上采样层
        self.up_blocks = nn.ModuleList()
        in_ch = mid_channels
        
        for i, mult in enumerate(reversed(channel_multipliers)):
            out_ch = base_channels * mult
            
            # 残差块
            res_blocks = nn.ModuleList()
            for _ in range(num_res_blocks + 1):  # +1因为跳跃连接
                res_blocks.append(ResidualBlock(in_ch, out_ch, dropout))
                in_ch = out_ch
            
            # 注意力块
            level = len(channel_multipliers) - 1 - i
            if 2**level in attention_resolutions:
                res_blocks.append(AttentionBlock(out_ch))
            
            self.up_blocks.append(res_blocks)
            
            # 上采样（除了最后一层）
            if i < len(channel_multipliers) - 1:
                self.up_blocks.append(
                    nn.ConvTranspose2d(out_ch, out_ch, 4, stride=2, padding=1)
                )
        
        # 输出层
        self.norm_out = nn.GroupNorm(32, base_channels)
        self.conv_out = nn.Conv2d(base_channels, out_channels, 3, padding=1)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: 潜在变量 [B, latent_channels, H', W']
            
        Returns:
            重构图像 [B, out_channels, H, W]
        """
        h = self.conv_in(z)
        
        # 中间层
        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)
        
        # 上采样
        for block in self.up_blocks:
            if isinstance(block, nn.ModuleList):
                for layer in block:
                    h = layer(h)
            else:
                h = block(h)
        
        # 输出
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        return h


class VAE(nn.Module):
    """完整的VAE模型"""
    
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 128,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16,),
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.encoder = VAEEncoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout
        )
        
        self.decoder = VAEDecoder(
            out_channels=in_channels,
            latent_channels=latent_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout
        )
        
        # 量化层（类似Stable Diffusion）
        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码"""
        mu, logvar = self.encoder(x)
        mu, logvar = torch.chunk(self.quant_conv(torch.cat([mu, logvar], dim=1)), 2, dim=1)
        return mu, logvar
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """解码"""
        z = self.post_quant_conv(z)
        return self.decoder(z)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: 输入图像 [B, C, H, W]
            return_dict: 是否返回字典格式
            
        Returns:
            重构结果字典
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        
        if return_dict:
            return {
                'reconstruction': recon,
                'mu': mu,
                'logvar': logvar,
                'latent': z
            }
        else:
            return recon, mu, logvar, z
    
    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """从先验分布采样"""
        # 假设潜在空间大小为64x64
        z = torch.randn(batch_size, self.encoder.latent_channels, 64, 64, device=device)
        return self.decode(z)


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 1e-6
) -> Dict[str, torch.Tensor]:
    """VAE损失函数"""
    
    # 重构损失（L1 + L2）
    l1_loss = F.l1_loss(recon, target, reduction='mean')
    l2_loss = F.mse_loss(recon, target, reduction='mean')
    recon_loss = l1_loss + l2_loss
    
    # KL散度损失
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / (target.numel())  # 标准化
    
    total_loss = recon_loss + kl_weight * kl_loss
    
    return {
        'total_loss': total_loss,
        'recon_loss': recon_loss,
        'kl_loss': kl_loss,
        'l1_loss': l1_loss,
        'l2_loss': l2_loss
    }


if __name__ == "__main__":
    # 测试VAE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vae = VAE(
        in_channels=3,
        latent_channels=4,
        base_channels=64,  # 减小以便测试
        channel_multipliers=(1, 2, 2, 4)
    ).to(device)
    
    # 测试输入
    test_images = torch.randn(2, 3, 256, 256).to(device)
    
    print("VAE模型测试:")
    print(f"参数量: {sum(p.numel() for p in vae.parameters()):,}")
    
    with torch.no_grad():
        # 前向传播
        results = vae(test_images)
        
        print(f"输入形状: {test_images.shape}")
        print(f"重构形状: {results['reconstruction'].shape}")
        print(f"潜在变量形状: {results['latent'].shape}")
        print(f"均值形状: {results['mu'].shape}")
        print(f"对数方差形状: {results['logvar'].shape}")
        
        # 测试损失函数
        loss_dict = vae_loss(
            results['reconstruction'],
            test_images,
            results['mu'], 
            results['logvar']
        )
        
        print("\n损失函数测试:")
        for key, value in loss_dict.items():
            print(f"  {key}: {value.item():.6f}")
    
    print("\nVAE模型测试完成！")