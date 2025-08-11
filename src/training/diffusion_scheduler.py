import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, Union
import math


class DDPMScheduler:
    """DDPM噪声调度器"""
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[np.ndarray] = None,
        variance_type: str = "fixed_small",
        clip_sample: bool = True,
        prediction_type: str = "epsilon"
    ):
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.variance_type = variance_type
        self.clip_sample = clip_sample
        self.prediction_type = prediction_type
        
        if trained_betas is not None:
            self.betas = torch.from_numpy(trained_betas).float()
        else:
            self.betas = self._get_beta_schedule()
        
        # 预计算常数
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
        
        # 计算后验方差
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
        # 后验均值系数
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
        
    def _get_beta_schedule(self) -> torch.Tensor:
        """获取beta调度"""
        if self.beta_schedule == "linear":
            return torch.linspace(self.beta_start, self.beta_end, self.num_train_timesteps)
        elif self.beta_schedule == "scaled_linear":
            return torch.linspace(
                self.beta_start**0.5, self.beta_end**0.5, self.num_train_timesteps
            ) ** 2
        elif self.beta_schedule == "cosine":
            return self._cosine_beta_schedule()
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")
    
    def _cosine_beta_schedule(self, s: float = 0.008) -> torch.Tensor:
        """余弦beta调度"""
        steps = self.num_train_timesteps + 1
        x = torch.linspace(0, self.num_train_timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.num_train_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """向样本添加噪声"""
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)
        
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """单步去噪"""
        
        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type == "learned":
            model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
        else:
            predicted_variance = None
        
        # 1. 计算alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod_prev[timestep] if timestep > 0 else torch.ones_like(alpha_prod_t)
        beta_prod_t = 1 - alpha_prod_t
        
        # 2. 计算预测的原始样本
        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t**(0.5) * model_output) / alpha_prod_t**(0.5)
        elif self.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        else:
            raise ValueError(f"prediction_type {self.prediction_type} not supported")
        
        # 3. 裁剪预测的x_0
        if self.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
        
        # 4. 计算方差
        if self.variance_type == "fixed_small":
            variance = self.posterior_variance[timestep]
        elif self.variance_type == "fixed_small_log":
            variance = torch.log(torch.clamp(self.posterior_variance[timestep], min=1e-20))
        elif self.variance_type == "learned":
            variance = predicted_variance
        
        # 5. 计算后验均值
        pred_sample_coeff = (alpha_prod_t_prev**(0.5) * self.betas[timestep]) / beta_prod_t
        current_sample_coeff = self.alphas[timestep]**(0.5) * (1 - alpha_prod_t_prev) / beta_prod_t
        
        pred_prev_sample = pred_sample_coeff * pred_original_sample + current_sample_coeff * sample
        
        # 6. 添加噪声
        if timestep > 0:
            device = model_output.device
            if predicted_variance is not None:
                noise = torch.randn_like(model_output, generator=generator, device=device)
                if self.variance_type == "fixed_small_log":
                    noise = noise * torch.exp(0.5 * variance)
                else:
                    noise = noise * variance**(0.5)
            else:
                noise = torch.randn_like(model_output, generator=generator, device=device)
                noise = noise * self.posterior_variance[timestep]**(0.5)
                
            pred_prev_sample = pred_prev_sample + noise
        
        if return_dict:
            return {
                'prev_sample': pred_prev_sample,
                'pred_original_sample': pred_original_sample
            }
        else:
            return pred_prev_sample
    
    def get_velocity(
        self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """计算velocity prediction target"""
        alphas_cumprod = self.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        timesteps = timesteps.to(sample.device)
        
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity


class DDIMScheduler:
    """DDIM采样器"""
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[np.ndarray] = None,
        clip_sample: bool = True,
        set_alpha_to_one: bool = True,
        prediction_type: str = "epsilon"
    ):
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.clip_sample = clip_sample
        self.set_alpha_to_one = set_alpha_to_one
        
        if trained_betas is not None:
            self.betas = torch.from_numpy(trained_betas).float()
        else:
            self.betas = self._get_beta_schedule(beta_start, beta_end, beta_schedule)
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # 标准的DDIM参数
        self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]
        
        # 设置推理时间步
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy())
    
    def _get_beta_schedule(self, beta_start: float, beta_end: float, beta_schedule: str) -> torch.Tensor:
        """获取beta调度"""
        if beta_schedule == "linear":
            return torch.linspace(beta_start, beta_end, self.num_train_timesteps)
        elif beta_schedule == "scaled_linear":
            return torch.linspace(beta_start**0.5, beta_end**0.5, self.num_train_timesteps) ** 2
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
    
    def set_timesteps(self, num_inference_steps: int):
        """设置推理时间步"""
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """DDIM单步去噪"""
        
        # 获取前一个时间步
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps
        
        # 计算alphas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        
        beta_prod_t = 1 - alpha_prod_t
        
        # 计算预测的原始样本
        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t**(0.5) * model_output) / alpha_prod_t**(0.5)
        elif self.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        
        # 裁剪预测的x_0
        if self.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
        
        # 计算方差
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance**(0.5)
        
        if use_clipped_model_output:
            model_output = (sample - alpha_prod_t**(0.5) * pred_original_sample) / beta_prod_t**(0.5)
        
        # 计算预测的前一个样本
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2)**(0.5) * model_output
        pred_prev_sample = alpha_prod_t_prev**(0.5) * pred_original_sample + pred_sample_direction
        
        if eta > 0:
            device = model_output.device
            noise = torch.randn_like(model_output, generator=generator, device=device)
            pred_prev_sample = pred_prev_sample + std_dev_t * noise
        
        if return_dict:
            return {
                'prev_sample': pred_prev_sample,
                'pred_original_sample': pred_original_sample
            }
        else:
            return pred_prev_sample
    
    def _get_variance(self, timestep: int, prev_timestep: int) -> torch.Tensor:
        """计算方差"""
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """向样本添加噪声（与DDPM相同）"""
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)
        
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples


if __name__ == "__main__":
    # 测试调度器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 测试DDPM调度器
    ddpm_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="cosine")
    print("DDPM调度器测试:")
    print(f"训练时间步数: {ddpm_scheduler.num_train_timesteps}")
    print(f"Beta范围: {ddpm_scheduler.betas.min():.6f} - {ddpm_scheduler.betas.max():.6f}")
    
    # 测试添加噪声
    original = torch.randn(2, 4, 64, 64).to(device)
    noise = torch.randn_like(original)
    timesteps = torch.randint(0, 1000, (2,)).to(device)
    
    noisy = ddpm_scheduler.add_noise(original, noise, timesteps)
    print(f"原始样本形状: {original.shape}")
    print(f"噪声样本形状: {noisy.shape}")
    
    # 测试DDIM调度器
    ddim_scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    ddim_scheduler.set_timesteps(50)  # 50步推理
    
    print(f"\nDDIM调度器测试:")
    print(f"推理时间步数: {ddim_scheduler.num_inference_steps}")
    print(f"时间步: {ddim_scheduler.timesteps[:10]}...")  # 显示前10个时间步
    
    print("\n调度器测试完成！")