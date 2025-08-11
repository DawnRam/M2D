from dataclasses import dataclass
from typing import Optional, Tuple
import os

# 尝试导入路径配置
try:
    from .model_paths import DATA_ROOT, OUTPUT_DIRS, USE_VIT_SUBSTITUTE
    _use_custom_paths = True
    print("✓ 使用自定义路径配置")
except ImportError:
    print("⚠ 未找到model_paths配置，使用默认路径")
    _use_custom_paths = False
    DATA_ROOT = "./data/ISIC"
    OUTPUT_DIRS = {
        "checkpoints": "./checkpoints",
        "outputs": "./outputs",
        "logs": "./logs"
    }
    USE_VIT_SUBSTITUTE = True


@dataclass
class DataConfig:
    """数据相关配置"""
    data_root: str = DATA_ROOT
    image_size: int = 224
    batch_size: int = 16
    num_workers: int = 4
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    augmentation: bool = True
    

@dataclass 
class ModelConfig:
    """模型相关配置"""
    # PanDerm配置
    panderm_model: str = "panderm-large"  # 或 "panderm-base"
    panderm_freeze: bool = True
    feature_dim: int = 768  # PanDerm特征维度
    
    # VAE配置
    vae_type: str = "sd-vae"
    vae_channels: int = 4
    vae_latent_size: int = 64
    
    # Diffusion UNet配置
    unet_channels: int = 320
    unet_layers: int = 4
    attention_heads: int = 8
    time_embed_dim: int = 1024
    
    # Feature Fusion配置
    fusion_type: str = "cross_attention"  # "concat", "add", "cross_attention"
    fusion_layers: int = 2
    

@dataclass
class TrainingConfig:
    """训练相关配置"""
    # 基础训练参数
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 1000
    
    # 损失函数权重
    alpha_diffusion: float = 1.0      # 扩散损失权重
    beta_reconstruction: float = 0.5   # 重构损失权重  
    gamma_alignment: float = 0.3       # 对齐损失权重
    delta_perceptual: float = 0.2      # 感知损失权重
    
    # Diffusion参数
    num_diffusion_steps: int = 1000
    noise_schedule: str = "cosine"     # "linear", "cosine"
    
    # 保存和日志
    save_every: int = 10
    log_every: int = 100
    eval_every: int = 5
    

@dataclass
class Config:
    """总体配置"""
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    
    # 设备和加速
    device: str = "cuda"
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    
    # 实验管理
    experiment_name: str = "panderm_diffusion"
    output_dir: str = OUTPUT_DIRS["outputs"]
    checkpoint_dir: str = OUTPUT_DIRS["checkpoints"] 
    log_dir: str = OUTPUT_DIRS["logs"]
    
    # 随机种子
    seed: int = 42
    
    # 模型替代配置
    use_vit_substitute: bool = USE_VIT_SUBSTITUTE