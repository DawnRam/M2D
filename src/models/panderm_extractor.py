import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import timm
from transformers import AutoModel, AutoConfig
import os
import sys
import math
import warnings
from scipy import interpolate

# 添加配置路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from configs.model_paths import PANDERM_MODEL_PATHS, USE_VIT_SUBSTITUTE
except ImportError:
    PANDERM_MODEL_PATHS = {}
    USE_VIT_SUBSTITUTE = True

# 导入PanDerm模型实现
from .panderm_model import panderm_base_patch16_224_finetune, load_state_dict


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
        
        # 尝试加载真实的PanDerm模型
        panderm_path = PANDERM_MODEL_PATHS.get(model_name)
        
        if not USE_VIT_SUBSTITUTE and panderm_path and os.path.exists(panderm_path):
            print(f"✓ 加载PanDerm预训练模型: {panderm_path}")
            
            # 使用REPA-E的PanDerm模型架构
            try:
                model_args = {
                    'pretrained': False,
                    'num_classes': 2,
                    'drop_rate': 0.0,
                    'drop_path_rate': 0.1,
                    'attn_drop_rate': 0.0,
                    'drop_block_rate': None,
                    'use_mean_pooling': True,
                    'init_scale': 0.001,
                    'use_rel_pos_bias': True,
                    'init_values': 0.1,
                    'lin_probe': False,
                    'patch_size': 16,
                    'img_size': 256  # 明确设置256像素输入
                }
                
                self.backbone = panderm_base_patch16_224_finetune(**model_args)
                backbone_dim = 768  # PanDerm base的嵌入维度
                
                # 加载预训练权重
                checkpoint_model = torch.load(panderm_path, map_location='cpu')
                
                # 处理权重
                state_dict = self.backbone.state_dict()
                
                # 处理分类头权重（如果尺寸不匹配则删除）
                for k in ['head.weight', 'head.bias']:
                    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                        del checkpoint_model[k]
                
                # 处理相对位置编码
                if self.backbone.use_rel_pos_bias and "rel_pos_bias.relative_position_bias_table" in checkpoint_model:
                    num_layers = self.backbone.get_num_layers()
                    rel_pos_bias = checkpoint_model["rel_pos_bias.relative_position_bias_table"]
                    for i in range(num_layers):
                        checkpoint_model[f"blocks.{i}.attn.relative_position_bias_table"] = rel_pos_bias.clone()
                    checkpoint_model.pop("rel_pos_bias.relative_position_bias_table")
                
                # 处理位置编码
                self._process_position_embedding(self.backbone, checkpoint_model, model_args)
                
                # 加载处理后的权重
                load_state_dict(self.backbone, checkpoint_model)
                
                print(f"✓ 成功加载PanDerm预训练权重")
                
            except Exception as e:
                print(f"⚠ 加载PanDerm权重失败: {e}")
                print("  回退到ViT替代模型")
                # 回退到ViT
                self.backbone = timm.create_model(
                    'vit_base_patch16_224',
                    pretrained=True,
                    num_classes=0,
                    global_pool='',
                    img_size=256
                )
                backbone_dim = 768
        else:
            # 使用ViT作为替代
            if not USE_VIT_SUBSTITUTE:
                print(f"⚠ PanDerm模型文件不存在: {panderm_path}")
            print("  使用预训练的ViT模型作为替代")
            
            if "large" in model_name.lower():
                self.backbone = timm.create_model(
                    'vit_large_patch16_224', 
                    pretrained=True,  # 使用ImageNet预训练权重
                    num_classes=0,
                    global_pool='',
                    img_size=256  # 设置输入图像尺寸为256
                )
                backbone_dim = 1024
            else:
                self.backbone = timm.create_model(
                    'vit_base_patch16_224',
                    pretrained=True,  # 使用ImageNet预训练权重
                    num_classes=0,
                    global_pool='',
                    img_size=256  # 设置输入图像尺寸为256
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
        return_multi_scale: bool = False,
        return_repa_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: 输入图像 [B, C, H, W]
            return_multi_scale: 是否返回多尺度特征
            return_repa_features: 是否返回REPA对齐用的特征列表
            
        Returns:
            特征字典
        """
        B = x.shape[0]
        
        if return_repa_features:
            # 返回用于REPA对齐的多层特征列表
            features = {}
            feature_list = []
            
            # 前向传播并收集中间特征
            x_embed = self.backbone.patch_embed(x)  # [B, N, D]
            x_embed = self.backbone._pos_embed(x_embed)
            
            # 收集每一层的空间特征用于对齐
            for i, block in enumerate(self.backbone.blocks):
                x_embed = block(x_embed)
                
                # 收集指定层的特征用于REPA对齐
                if self.output_layers and i in self.output_layers:
                    feature_list.append(x_embed.clone())  # [B, N, D] 保持空间特征
                elif not self.output_layers and i % 3 == 0:  # 如果没有指定层，每3层收集一次
                    feature_list.append(x_embed.clone())
            
            # 最终特征
            x_embed = self.backbone.norm(x_embed)
            feature_list.append(x_embed.clone())  # 添加最终的标准化特征
            
            # 全局特征
            global_feature = x_embed.mean(dim=1)  # [B, D]
            global_feature = self.feature_projection(global_feature)
            
            features['global'] = global_feature
            features['spatial'] = x_embed
            features['repa_features'] = feature_list  # 用于REPA对齐的多层特征列表
            
            return features
            
        elif return_multi_scale and self.output_layers:
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
    
    def _process_position_embedding(self, encoder, checkpoint_model, model_args):
        """处理位置编码的辅助函数"""
        all_keys = list(checkpoint_model.keys())
        for key in all_keys:
            if "relative_position_index" in key:
                checkpoint_model.pop(key)
            
            if "relative_position_bias_table" in key and model_args['use_rel_pos_bias']:
                rel_pos_bias = checkpoint_model[key]
                src_num_pos, num_attn_heads = rel_pos_bias.size()
                dst_num_pos, _ = encoder.state_dict()[key].size()
                dst_patch_shape = encoder.patch_embed.patch_shape
                
                if dst_patch_shape[0] != dst_patch_shape[1]:
                    raise NotImplementedError()
                
                num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
                src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
                dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
                
                if src_size != dst_size:
                    checkpoint_model[key] = self._interpolate_rel_pos_bias(rel_pos_bias, src_size, dst_size, num_extra_tokens)

    def _interpolate_rel_pos_bias(self, rel_pos_bias, src_size, dst_size, num_extra_tokens):
        """插值相对位置偏置的辅助函数"""
        import numpy as np
        
        extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
        rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]
        
        # 几何级数计算
        def geometric_progression(a, r, n):
            return a * (1.0 - r ** n) / (1.0 - r)
        
        left, right = 1.01, 1.5
        while right - left > 1e-6:
            q = (left + right) / 2.0
            gp = geometric_progression(1, q, src_size // 2)
            if gp > dst_size // 2:
                right = q
            else:
                left = q
        
        # 生成位置索引
        dis = []
        cur = 1
        for i in range(src_size // 2):
            dis.append(cur)
            cur += q ** (i + 1)
        
        r_ids = [-_ for _ in reversed(dis)]
        x = r_ids + [0] + dis
        y = r_ids + [0] + dis
        
        t = dst_size // 2.0
        dx = np.arange(-t, t + 0.1, 1.0)
        dy = np.arange(-t, t + 0.1, 1.0)
        
        # 双三次插值
        all_rel_pos_bias = []
        for i in range(rel_pos_bias.size(1)):
            z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
            f = interpolate.interp2d(x, y, z, kind='cubic')
            all_rel_pos_bias.append(
                torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(rel_pos_bias.device))
        
        rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
        new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
        return new_rel_pos_bias


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
        # 维度检查和修复
        if query.dim() == 2:
            # 如果query是[B, D]，扩展为[B, 1, D]
            query = query.unsqueeze(1)
        elif query.dim() == 1:
            # 如果query是[D]，扩展为[1, 1, D]
            query = query.unsqueeze(0).unsqueeze(0)
            
        if key_value.dim() == 2:
            # 如果key_value是[B, D]，扩展为[B, 1, D]
            key_value = key_value.unsqueeze(1)
        elif key_value.dim() == 1:
            # 如果key_value是[D]，扩展为[1, 1, D]
            key_value = key_value.unsqueeze(0).unsqueeze(0)
        
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