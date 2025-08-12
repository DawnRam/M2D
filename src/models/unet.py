import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import math
from .panderm_extractor import FeatureFusionModule


class TimeEmbedding(nn.Module):
    """时间步嵌入"""
    
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        
        self.dim = dim
        self.max_period = max_period
        
        # 时间步投影
        self.time_proj = nn.Sequential(
            nn.Linear(dim // 4, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: [B,] 时间步
            
        Returns:
            时间嵌入 [B, dim]
        """
        half = self.dim // 8
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, dtype=torch.float32) / half
        ).to(timesteps.device)
        
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if self.dim % 8:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        
        return self.time_proj(embedding)


class ResnetBlock(nn.Module):
    """ResNet块"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        use_conv_shortcut: bool = False
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = use_conv_shortcut
        
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        # 时间嵌入投影
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels)
        )
        
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # 残差连接
        if in_channels != out_channels:
            if use_conv_shortcut:
                self.shortcut = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            else:
                self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_embed: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # 添加时间嵌入
        h = h + self.time_proj(time_embed)[:, :, None, None]
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)


class SpatialTransformer(nn.Module):
    """空间Transformer（支持cross-attention）"""
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        head_dim: int = 64,
        context_dim: Optional[int] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        
        self.norm = nn.GroupNorm(32, channels)
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-attention（如果提供context）
        if context_dim is not None:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=channels,
                num_heads=num_heads,
                kdim=context_dim,
                vdim=context_dim,
                dropout=dropout,
                batch_first=True
            )
        else:
            self.cross_attn = None
        
        # Feed Forward
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels)
        )
        
        self.norm_self = nn.LayerNorm(channels)
        self.norm_cross = nn.LayerNorm(channels) if self.cross_attn else None
        self.norm_ff = nn.LayerNorm(channels)
        
    def forward(
        self, 
        x: torch.Tensor, 
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
            context: [B, N, context_dim] 可选的context特征
        """
        B, C, H, W = x.shape
        
        # 转换为序列格式
        h = self.norm(x)
        h = h.view(B, C, H * W).transpose(1, 2)  # [B, H*W, C]
        
        # Self-attention
        attn_out, _ = self.self_attn(h, h, h)
        h = h + attn_out
        h = self.norm_self(h)
        
        # Cross-attention（如果有context）
        if self.cross_attn is not None and context is not None:
            # 确保context张量的维度正确
            if context.dim() == 2:
                # 如果context是[B, D]，需要扩展为[B, 1, D]
                context = context.unsqueeze(1)
            elif context.dim() == 1:
                # 如果context是[D]，需要扩展为[1, 1, D]然后广播
                context = context.unsqueeze(0).unsqueeze(0)
                context = context.expand(B, 1, -1)
            
            # 检查维度是否匹配
            if context.size(0) != B:
                raise ValueError(f"Context batch size {context.size(0)} doesn't match input batch size {B}")
            
            cross_out, _ = self.cross_attn(h, context, context)
            h = h + cross_out
            if self.norm_cross is not None:
                h = self.norm_cross(h)
        
        # Feed Forward
        ff_out = self.ff(h)
        h = h + ff_out
        h = self.norm_ff(h)
        
        # 转换回空间格式
        h = h.transpose(1, 2).view(B, C, H, W)
        
        return x + h


class DownBlock(nn.Module):
    """下采样块"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        num_layers: int = 2,
        downsample: bool = True,
        use_attention: bool = False,
        num_heads: int = 8,
        context_dim: Optional[int] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.downsample = downsample
        self.use_attention = use_attention
        
        # ResNet层
        self.resnets = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            self.resnets.append(ResnetBlock(
                in_ch, out_channels, time_embed_dim, dropout
            ))
        
        # 注意力层
        if use_attention:
            self.attentions = nn.ModuleList([
                SpatialTransformer(
                    out_channels, num_heads, out_channels // num_heads,
                    context_dim, dropout
                ) for _ in range(num_layers)
            ])
        else:
            self.attentions = None
        
        # 下采样
        if downsample:
            self.downsample_conv = nn.Conv2d(
                out_channels, out_channels, 3, stride=2, padding=1
            )
    
    def forward(
        self, 
        x: torch.Tensor, 
        time_embed: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        
        outputs = []
        
        for i, resnet in enumerate(self.resnets):
            x = resnet(x, time_embed)
            
            if self.use_attention:
                x = self.attentions[i](x, context)
            
            outputs.append(x)
        
        if self.downsample:
            x = self.downsample_conv(x)
            outputs.append(x)
        
        return x, outputs


class UpBlock(nn.Module):
    """上采样块"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        time_embed_dim: int,
        num_layers: int = 3,
        upsample: bool = True,
        use_attention: bool = False,
        num_heads: int = 8,
        context_dim: Optional[int] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.upsample = upsample
        self.use_attention = use_attention
        
        # ResNet层（包括跳跃连接）
        self.resnets = nn.ModuleList()
        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            resnet_in_channels += res_skip_channels
            
            self.resnets.append(ResnetBlock(
                resnet_in_channels, out_channels, time_embed_dim, dropout
            ))
        
        # 注意力层
        if use_attention:
            self.attentions = nn.ModuleList([
                SpatialTransformer(
                    out_channels, num_heads, out_channels // num_heads,
                    context_dim, dropout
                ) for _ in range(num_layers)
            ])
        else:
            self.attentions = None
        
        # 上采样
        if upsample:
            self.upsample_conv = nn.ConvTranspose2d(
                out_channels, out_channels, 4, stride=2, padding=1
            )
    
    def forward(
        self,
        x: torch.Tensor,
        res_samples: List[torch.Tensor],
        time_embed: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        for i, resnet in enumerate(self.resnets):
            # 添加跳跃连接
            res_x = res_samples[-1]
            res_samples = res_samples[:-1]
            x = torch.cat([x, res_x], dim=1)
            
            x = resnet(x, time_embed)
            
            if self.use_attention:
                x = self.attentions[i](x, context)
        
        if self.upsample:
            x = self.upsample_conv(x)
        
        return x


class UNet2D(nn.Module):
    """改进的UNet2D模型"""
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        model_channels: int = 320,
        time_embed_dim: int = 1024,
        attention_resolutions: Tuple[int, ...] = (4, 2, 1),
        channel_mult: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        num_heads: int = 8,
        context_dim: Optional[int] = 768,  # PanDerm特征维度
        use_feature_fusion: bool = True,
        fusion_type: str = "cross_attention"
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.time_embed_dim = time_embed_dim
        self.use_feature_fusion = use_feature_fusion
        
        # 时间嵌入
        self.time_embed = TimeEmbedding(time_embed_dim)
        
        # 特征融合模块
        if use_feature_fusion and context_dim is not None:
            self.feature_fusion = FeatureFusionModule(
                panderm_dim=context_dim,
                latent_dim=model_channels,
                fusion_type=fusion_type,
                num_heads=8,
                num_layers=2
            )
        else:
            self.feature_fusion = None
        
        # 输入投影
        self.conv_in = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # 计算各层通道数
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        
        # 下采样块
        self.down_blocks = nn.ModuleList()
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            
            for _ in range(num_res_blocks):
                use_attention = ds in attention_resolutions
                
                block = DownBlock(
                    in_channels=ch,
                    out_channels=out_ch,
                    time_embed_dim=time_embed_dim,
                    num_layers=1,
                    downsample=False,
                    use_attention=use_attention,
                    num_heads=num_heads,
                    context_dim=context_dim,
                    dropout=dropout
                )
                
                self.down_blocks.append(block)
                ch = out_ch
                input_block_chans.append(ch)
            
            # 下采样（除了最后一层）
            if level != len(channel_mult) - 1:
                block = DownBlock(
                    in_channels=ch,
                    out_channels=ch,
                    time_embed_dim=time_embed_dim,
                    num_layers=1,
                    downsample=True,
                    use_attention=False,
                    dropout=dropout
                )
                
                self.down_blocks.append(block)
                input_block_chans.append(ch)
                ds *= 2
        
        # 中间块
        mid_ch = ch
        self.mid_block1 = ResnetBlock(
            mid_ch, mid_ch, time_embed_dim, dropout
        )
        self.mid_attn = SpatialTransformer(
            mid_ch, num_heads, mid_ch // num_heads, context_dim, dropout
        )
        self.mid_block2 = ResnetBlock(
            mid_ch, mid_ch, time_embed_dim, dropout
        )
        
        # 上采样块
        self.up_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            out_ch = model_channels * mult
            
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                use_attention = ds in attention_resolutions
                
                block = UpBlock(
                    in_channels=ch,
                    out_channels=out_ch,
                    prev_output_channel=ich,
                    time_embed_dim=time_embed_dim,
                    num_layers=1,
                    upsample=False,
                    use_attention=use_attention,
                    num_heads=num_heads,
                    context_dim=context_dim,
                    dropout=dropout
                )
                
                self.up_blocks.append(block)
                ch = out_ch
            
            # 上采样
            if level != 0:
                block = UpBlock(
                    in_channels=ch,
                    out_channels=ch,
                    prev_output_channel=0,
                    time_embed_dim=time_embed_dim,
                    num_layers=1,
                    upsample=True,
                    use_attention=False,
                    dropout=dropout
                )
                
                self.up_blocks.append(block)
                ds //= 2
        
        # 输出层
        self.norm_out = nn.GroupNorm(32, model_channels)
        self.conv_out = nn.Conv2d(model_channels, out_channels, 3, padding=1)
        
        # 零初始化输出权重
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        panderm_features: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, in_channels, H, W] 输入噪声图像
            timesteps: [B,] 时间步
            panderm_features: [B, feature_dim] PanDerm特征
            return_dict: 是否返回字典
            
        Returns:
            预测的噪声
        """
        
        # 时间嵌入
        time_embed = self.time_embed(timesteps)
        
        # 输入投影
        h = self.conv_in(x)
        
        # 特征融合（如果启用）
        if self.use_feature_fusion and panderm_features is not None:
            if panderm_features.dim() == 2:  # [B, D]
                panderm_context = panderm_features.unsqueeze(1)  # [B, 1, D]
            else:
                panderm_context = panderm_features  # [B, N, D]
        else:
            panderm_context = None
        
        # 下采样
        down_samples = []
        for block in self.down_blocks:
            h, block_samples = block(h, time_embed, panderm_context)
            down_samples.extend(block_samples)
        
        # 中间层
        h = self.mid_block1(h, time_embed)
        h = self.mid_attn(h, panderm_context)
        h = self.mid_block2(h, time_embed)
        
        # 上采样
        for block in self.up_blocks:
            samples_for_block = down_samples[-block.num_layers:]
            down_samples = down_samples[:-block.num_layers]
            h = block(h, samples_for_block, time_embed, panderm_context)
        
        # 输出
        h = self.norm_out(h)
        h = F.silu(h)
        noise_pred = self.conv_out(h)
        
        if return_dict:
            return {'sample': noise_pred}
        else:
            return noise_pred


if __name__ == "__main__":
    # 测试UNet模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    unet = UNet2D(
        in_channels=4,
        out_channels=4,
        model_channels=64,  # 减小以便测试
        channel_mult=(1, 2, 2),
        num_res_blocks=1,
        context_dim=768,
        use_feature_fusion=True
    ).to(device)
    
    print("UNet2D模型测试:")
    print(f"参数量: {sum(p.numel() for p in unet.parameters()):,}")
    
    # 测试输入
    batch_size = 2
    x = torch.randn(batch_size, 4, 64, 64).to(device)
    timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
    panderm_features = torch.randn(batch_size, 768).to(device)
    
    with torch.no_grad():
        output = unet(x, timesteps, panderm_features)
        
        print(f"输入形状: {x.shape}")
        print(f"时间步形状: {timesteps.shape}")
        print(f"PanDerm特征形状: {panderm_features.shape}")
        print(f"输出形状: {output['sample'].shape}")
    
    print("UNet2D模型测试完成！")