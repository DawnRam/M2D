import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import timm
from transformers import AutoModel, AutoConfig


class PanDermFeatureExtractor(nn.Module):
    """PanDerm特征提取器"""
    
    def __init__(
        self,
        model_name: str = "panderm-large",
        freeze_backbone: bool = True,
        feature_dim: int = 768,
        output_layers: Optional[Tuple[int, ...]] = None
    ):
        """
        Args:
            model_name: PanDerm模型名称
            freeze_backbone: 是否冻结backbone权重
            feature_dim: 输出特征维度
            output_layers: 要输出的层索引（多尺度特征）
        """
        super().__init__()
        
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.feature_dim = feature_dim
        self.output_layers = output_layers
        
        # 加载预训练模型（这里使用ViT作为替代实现）
        # 实际项目中需要加载真正的PanDerm模型权重
        if "large" in model_name.lower():
            self.backbone = timm.create_model(
                'vit_large_patch16_224', 
                pretrained=True,
                num_classes=0,  # 移除分类头
                global_pool=''  # 移除全局池化
            )
            backbone_dim = 1024
        else:
            self.backbone = timm.create_model(
                'vit_base_patch16_224',
                pretrained=True, 
                num_classes=0,
                global_pool=''
            )
            backbone_dim = 768
        
        # 特征投影层
        if backbone_dim != feature_dim:
            self.feature_projection = nn.Linear(backbone_dim, feature_dim)
        else:
            self.feature_projection = nn.Identity()
        
        # 多尺度特征提取
        if output_layers:
            self.multi_scale_projections = nn.ModuleDict()
            for layer_idx in output_layers:
                self.multi_scale_projections[f'layer_{layer_idx}'] = nn.Linear(
                    backbone_dim, feature_dim
                )
        
        # 冻结backbone
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """冻结backbone参数"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def forward(
        self, 
        x: torch.Tensor,
        return_multi_scale: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: 输入图像 [B, C, H, W]
            return_multi_scale: 是否返回多尺度特征
            
        Returns:
            特征字典
        """
        B = x.shape[0]
        
        if return_multi_scale and self.output_layers:
            # 获取多尺度特征
            features = {}
            
            # 前向传播并收集中间特征
            x_embed = self.backbone.patch_embed(x)  # [B, N, D]
            x_embed = self.backbone._pos_embed(x_embed)
            
            for i, block in enumerate(self.backbone.blocks):
                x_embed = block(x_embed)
                
                if i in self.output_layers:
                    # 全局平均池化
                    pooled_feature = x_embed.mean(dim=1)  # [B, D]
                    projected_feature = self.multi_scale_projections[f'layer_{i}'](pooled_feature)
                    features[f'layer_{i}'] = projected_feature
            
            # 最终特征
            x_embed = self.backbone.norm(x_embed)
            global_feature = x_embed.mean(dim=1)  # [B, D]
            features['global'] = self.feature_projection(global_feature)
            
            return features
        else:
            # 标准前向传播
            x_embed = self.backbone.forward_features(x)  # [B, N, D]
            
            # 全局特征（CLS token或平均池化）
            if hasattr(self.backbone, 'forward_head'):
                global_feature = self.backbone.forward_head(x_embed, pre_logits=True)
            else:
                global_feature = x_embed.mean(dim=1)  # [B, D]
            
            # 投影到目标维度
            global_feature = self.feature_projection(global_feature)
            
            return {
                'global': global_feature,  # [B, feature_dim]
                'spatial': x_embed,        # [B, N, D] 空间特征用于cross-attention
            }


class FeatureFusionModule(nn.Module):
    """特征融合模块"""
    
    def __init__(
        self,
        panderm_dim: int = 768,
        latent_dim: int = 320,
        fusion_type: str = "cross_attention",
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            panderm_dim: PanDerm特征维度
            latent_dim: 潜在空间维度
            fusion_type: 融合方式 ("concat", "add", "cross_attention", "adaptive")
            num_heads: attention头数
            num_layers: 融合层数
            dropout: dropout率
        """
        super().__init__()
        
        self.panderm_dim = panderm_dim
        self.latent_dim = latent_dim
        self.fusion_type = fusion_type
        self.num_heads = num_heads
        
        if fusion_type == "concat":
            self.fusion_layers = nn.Sequential(
                nn.Linear(panderm_dim + latent_dim, latent_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(latent_dim * 2, latent_dim),
                nn.LayerNorm(latent_dim)
            )
            
        elif fusion_type == "add":
            self.panderm_projection = nn.Linear(panderm_dim, latent_dim)
            self.fusion_norm = nn.LayerNorm(latent_dim)
            
        elif fusion_type == "cross_attention":
            # Cross-attention融合
            self.cross_attention_layers = nn.ModuleList([
                CrossAttentionBlock(
                    query_dim=latent_dim,
                    key_value_dim=panderm_dim,
                    num_heads=num_heads,
                    dropout=dropout
                ) for _ in range(num_layers)
            ])
            
        elif fusion_type == "adaptive":
            # 自适应加权融合
            self.panderm_projection = nn.Linear(panderm_dim, latent_dim)
            self.gate_network = nn.Sequential(
                nn.Linear(panderm_dim + latent_dim, latent_dim // 2),
                nn.GELU(),
                nn.Linear(latent_dim // 2, 1),
                nn.Sigmoid()
            )
            self.fusion_norm = nn.LayerNorm(latent_dim)
    
    def forward(
        self,
        latent_features: torch.Tensor,
        panderm_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            latent_features: 潜在空间特征 [B, D_latent] 或 [B, H*W, D_latent]
            panderm_features: PanDerm特征 [B, D_panderm] 或 [B, N, D_panderm]
            attention_mask: 注意力掩码
            
        Returns:
            融合后的特征
        """
        if self.fusion_type == "concat":
            # 确保维度匹配
            if latent_features.dim() == 3:  # spatial features
                B, HW, D = latent_features.shape
                panderm_global = panderm_features.mean(dim=1) if panderm_features.dim() == 3 else panderm_features
                panderm_expanded = panderm_global.unsqueeze(1).expand(B, HW, self.panderm_dim)
                concat_features = torch.cat([latent_features, panderm_expanded], dim=-1)
                return self.fusion_layers(concat_features)
            else:
                panderm_global = panderm_features.mean(dim=1) if panderm_features.dim() == 3 else panderm_features
                concat_features = torch.cat([latent_features, panderm_global], dim=-1)
                return self.fusion_layers(concat_features)
                
        elif self.fusion_type == "add":
            panderm_projected = self.panderm_projection(panderm_features)
            if panderm_projected.dim() == 2 and latent_features.dim() == 3:
                panderm_projected = panderm_projected.unsqueeze(1)
            fused = latent_features + panderm_projected
            return self.fusion_norm(fused)
            
        elif self.fusion_type == "cross_attention":
            fused_features = latent_features
            for layer in self.cross_attention_layers:
                fused_features = layer(
                    query=fused_features,
                    key_value=panderm_features,
                    attention_mask=attention_mask
                )
            return fused_features
            
        elif self.fusion_type == "adaptive":
            panderm_projected = self.panderm_projection(panderm_features)
            
            # 计算自适应权重
            if latent_features.dim() == 3 and panderm_features.dim() == 2:
                panderm_features_expanded = panderm_features.unsqueeze(1).expand_as(latent_features)
                gate_input = torch.cat([latent_features, panderm_features_expanded], dim=-1)
            else:
                gate_input = torch.cat([latent_features, panderm_features], dim=-1)
            
            gate_weights = self.gate_network(gate_input)
            
            # 自适应融合
            if panderm_projected.dim() == 2 and latent_features.dim() == 3:
                panderm_projected = panderm_projected.unsqueeze(1)
                
            fused = gate_weights * latent_features + (1 - gate_weights) * panderm_projected
            return self.fusion_norm(fused)


class CrossAttentionBlock(nn.Module):
    """Cross-Attention块"""
    
    def __init__(
        self,
        query_dim: int,
        key_value_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(key_value_dim, query_dim)
        self.v_proj = nn.Linear(key_value_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        
        # Feed Forward
        self.ff = nn.Sequential(
            nn.Linear(query_dim, query_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_dim * 4, query_dim)
        )
    
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: [B, N_q, D_q]
            key_value: [B, N_kv, D_kv]
            attention_mask: [B, N_q, N_kv]
        """
        B, N_q, D_q = query.shape
        N_kv = key_value.shape[1]
        
        # Multi-head attention
        q = self.q_proj(query).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention计算
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(attention_mask == 0, -1e9)
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)  # [B, H, N_q, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N_q, D_q)
        
        # 残差连接和LayerNorm
        query = query + self.dropout(self.out_proj(attn_output))
        query = self.norm1(query)
        
        # Feed Forward
        ff_output = self.ff(query)
        query = query + self.dropout(ff_output)
        query = self.norm2(query)
        
        return query


if __name__ == "__main__":
    # 测试特征提取器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 测试PanDerm特征提取器
    extractor = PanDermFeatureExtractor(
        model_name="panderm-large",
        freeze_backbone=True,
        feature_dim=768
    ).to(device)
    
    # 测试输入
    test_images = torch.randn(2, 3, 224, 224).to(device)
    
    with torch.no_grad():
        features = extractor(test_images)
        print("PanDerm特征提取器输出:")
        for key, value in features.items():
            print(f"  {key}: {value.shape}")
    
    # 测试特征融合模块
    fusion = FeatureFusionModule(
        panderm_dim=768,
        latent_dim=320,
        fusion_type="cross_attention"
    ).to(device)
    
    latent_features = torch.randn(2, 64*64, 320).to(device)  # 空间特征
    panderm_features = torch.randn(2, 197, 768).to(device)   # ViT特征
    
    with torch.no_grad():
        fused_features = fusion(latent_features, panderm_features)
        print(f"\n融合特征输出: {fused_features.shape}")
    
    print("\n特征提取和融合模块测试完成！")