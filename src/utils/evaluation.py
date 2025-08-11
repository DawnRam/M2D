import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from PIL import Image
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import cv2
from scipy import linalg
from torchvision import transforms
from torchvision.models import inception_v3
import lpips


class FIDCalculator:
    """Fréchet Inception Distance (FID) 计算器"""
    
    def __init__(self, device: torch.device):
        self.device = device
        
        # 加载预训练的Inception网络
        self.inception = inception_v3(pretrained=True, transform_input=False).to(device)
        self.inception.eval()
        
        # 移除最后的分类层，获取特征
        self.inception.fc = nn.Identity()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def get_features(self, images: torch.Tensor, batch_size: int = 50) -> np.ndarray:
        """提取图像特征"""
        
        features = []
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size]
                
                # 预处理
                if batch.shape[1] == 3:  # RGB图像
                    # 假设输入在[0,1]范围内
                    batch_processed = self.transform(batch)
                else:
                    raise ValueError("图像必须是3通道RGB格式")
                
                # 提取特征
                feat = self.inception(batch_processed.to(self.device))
                features.append(feat.cpu().numpy())
        
        return np.concatenate(features, axis=0)
    
    def calculate_fid(
        self, 
        real_images: torch.Tensor, 
        fake_images: torch.Tensor
    ) -> float:
        """计算FID分数"""
        
        # 提取特征
        real_features = self.get_features(real_images)
        fake_features = self.get_features(fake_images)
        
        # 计算均值和协方差
        mu_real, sigma_real = self._calculate_statistics(real_features)
        mu_fake, sigma_fake = self._calculate_statistics(fake_features)
        
        # 计算FID
        fid = self._calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
        
        return fid
    
    def _calculate_statistics(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算特征的统计量"""
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma
    
    def _calculate_frechet_distance(
        self, 
        mu1: np.ndarray, 
        sigma1: np.ndarray, 
        mu2: np.ndarray, 
        sigma2: np.ndarray
    ) -> float:
        """计算Fréchet距离"""
        
        diff = mu1 - mu2
        
        # 计算协方差矩阵的平方根
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        # 处理数值不稳定性
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
        
        return float(fid)


class ISCalculator:
    """Inception Score (IS) 计算器"""
    
    def __init__(self, device: torch.device):
        self.device = device
        
        # 加载预训练的Inception网络（保留分类层）
        self.inception = inception_v3(pretrained=True, transform_input=False).to(device)
        self.inception.eval()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def calculate_is(
        self, 
        images: torch.Tensor, 
        batch_size: int = 50, 
        splits: int = 10
    ) -> Tuple[float, float]:
        """计算Inception Score"""
        
        # 获取预测概率
        probs = self._get_predictions(images, batch_size)
        
        # 计算IS
        scores = []
        for i in range(splits):
            part = probs[i * (len(probs) // splits): (i + 1) * (len(probs) // splits)]
            kl_div = self._calculate_kl_divergence(part)
            scores.append(np.exp(kl_div))
        
        return np.mean(scores), np.std(scores)
    
    def _get_predictions(self, images: torch.Tensor, batch_size: int) -> np.ndarray:
        """获取分类预测"""
        
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size]
                batch_processed = self.transform(batch).to(self.device)
                
                pred = self.inception(batch_processed)
                pred = F.softmax(pred, dim=1)
                predictions.append(pred.cpu().numpy())
        
        return np.concatenate(predictions, axis=0)
    
    def _calculate_kl_divergence(self, probs: np.ndarray) -> float:
        """计算KL散度"""
        
        # 边际分布
        marginal = np.mean(probs, axis=0)
        
        # KL散度
        kl_div = 0.0
        for p in probs:
            kl_div += np.sum(p * np.log(p / marginal + 1e-8))
        
        return kl_div / len(probs)


class LPIPSCalculator:
    """Learned Perceptual Image Patch Similarity (LPIPS) 计算器"""
    
    def __init__(self, device: torch.device, network: str = 'alex'):
        self.device = device
        self.lpips_model = lpips.LPIPS(net=network).to(device)
    
    def calculate_lpips(
        self, 
        images1: torch.Tensor, 
        images2: torch.Tensor
    ) -> float:
        """计算LPIPS距离"""
        
        assert images1.shape == images2.shape, "图像尺寸必须相同"
        
        # 确保图像在[-1, 1]范围内
        if images1.max() <= 1.0 and images1.min() >= 0.0:
            images1 = images1 * 2.0 - 1.0
        if images2.max() <= 1.0 and images2.min() >= 0.0:
            images2 = images2 * 2.0 - 1.0
        
        with torch.no_grad():
            lpips_dist = self.lpips_model(
                images1.to(self.device), 
                images2.to(self.device)
            )
        
        return float(lpips_dist.mean().cpu())


class MedicalImageEvaluator:
    """医学图像特定评估器"""
    
    def __init__(self, device: torch.device):
        self.device = device
    
    def calculate_ssim(
        self, 
        images1: torch.Tensor, 
        images2: torch.Tensor
    ) -> float:
        """计算结构相似性指数 (SSIM)"""
        
        def _ssim_single(img1, img2):
            """单张图像SSIM计算"""
            # 转换为numpy
            if torch.is_tensor(img1):
                img1 = img1.cpu().numpy()
                img2 = img2.cpu().numpy()
            
            # 转换为灰度图像
            if len(img1.shape) == 3:
                img1 = cv2.cvtColor(img1.transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
                img2 = cv2.cvtColor(img2.transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
            
            # 计算SSIM
            from skimage.metrics import structural_similarity
            return structural_similarity(img1, img2, data_range=1.0)
        
        ssim_scores = []
        for i in range(len(images1)):
            ssim = _ssim_single(images1[i], images2[i])
            ssim_scores.append(ssim)
        
        return np.mean(ssim_scores)
    
    def calculate_psnr(
        self, 
        images1: torch.Tensor, 
        images2: torch.Tensor
    ) -> float:
        """计算峰值信噪比 (PSNR)"""
        
        mse = F.mse_loss(images1, images2, reduction='none')
        mse = mse.view(len(images1), -1).mean(dim=1)
        
        # 避免除以0
        mse = torch.clamp(mse, min=1e-8)
        
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        return float(psnr.mean())
    
    def calculate_feature_diversity(
        self, 
        features: torch.Tensor
    ) -> Dict[str, float]:
        """计算特征多样性"""
        
        # 特征标准化
        features_norm = F.normalize(features, p=2, dim=1)
        
        # 计算相似性矩阵
        similarity_matrix = torch.mm(features_norm, features_norm.t())
        
        # 去除对角线元素
        n = similarity_matrix.shape[0]
        mask = torch.eye(n, device=similarity_matrix.device).bool()
        similarities = similarity_matrix[~mask]
        
        # 多样性指标
        diversity_metrics = {
            'mean_similarity': float(similarities.mean()),
            'std_similarity': float(similarities.std()),
            'min_similarity': float(similarities.min()),
            'max_similarity': float(similarities.max()),
            'diversity_score': float(1.0 - similarities.mean())  # 1 - 平均相似度
        }
        
        return diversity_metrics
    
    def calculate_lesion_coverage(
        self, 
        generated_images: torch.Tensor,
        real_images: torch.Tensor,
        lesion_detector: Optional[nn.Module] = None
    ) -> Dict[str, float]:
        """计算病变覆盖度（需要病变检测模型）"""
        
        if lesion_detector is None:
            # 简化版本：基于颜色和纹理特征
            return self._simple_lesion_coverage(generated_images, real_images)
        
        # 使用专门的病变检测模型
        with torch.no_grad():
            real_lesions = lesion_detector(real_images)
            gen_lesions = lesion_detector(generated_images)
        
        coverage_metrics = {
            'lesion_detection_rate': float((gen_lesions > 0.5).float().mean()),
            'lesion_similarity': float(F.cosine_similarity(real_lesions, gen_lesions).mean())
        }
        
        return coverage_metrics
    
    def _simple_lesion_coverage(
        self, 
        generated_images: torch.Tensor,
        real_images: torch.Tensor
    ) -> Dict[str, float]:
        """简化的病变覆盖度计算"""
        
        def extract_color_features(images):
            """提取颜色特征"""
            # 转换为HSV空间
            features = []
            for img in images:
                img_np = img.cpu().numpy().transpose(1, 2, 0)
                img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
                
                # 计算颜色直方图
                hist_h = cv2.calcHist([img_hsv], [0], None, [50], [0, 180])
                hist_s = cv2.calcHist([img_hsv], [1], None, [50], [0, 256])
                hist_v = cv2.calcHist([img_hsv], [2], None, [50], [0, 256])
                
                hist = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
                features.append(hist)
            
            return np.array(features)
        
        # 提取特征
        real_features = extract_color_features(real_images)
        gen_features = extract_color_features(generated_images)
        
        # 计算特征相似性
        from scipy.spatial.distance import cosine
        similarities = []
        
        for gen_feat in gen_features:
            max_sim = 0
            for real_feat in real_features:
                sim = 1 - cosine(gen_feat, real_feat)
                max_sim = max(max_sim, sim)
            similarities.append(max_sim)
        
        return {
            'color_feature_similarity': float(np.mean(similarities)),
            'feature_coverage_rate': float(np.mean(np.array(similarities) > 0.7))
        }


class ComprehensiveEvaluator:
    """综合评估器"""
    
    def __init__(self, device: torch.device):
        self.device = device
        
        # 初始化各个评估器
        self.fid_calculator = FIDCalculator(device)
        self.is_calculator = ISCalculator(device)
        self.lpips_calculator = LPIPSCalculator(device)
        self.medical_evaluator = MedicalImageEvaluator(device)
    
    def evaluate_generation_quality(
        self,
        generated_images: torch.Tensor,
        real_images: torch.Tensor,
        panderm_features: Optional[torch.Tensor] = None,
        save_results: bool = True,
        output_dir: str = "./evaluation_results"
    ) -> Dict[str, Union[float, Dict]]:
        """综合评估生成质量"""
        
        print("开始综合评估...")
        
        results = {}
        
        # 1. FID Score
        print("计算FID...")
        try:
            fid_score = self.fid_calculator.calculate_fid(real_images, generated_images)
            results['fid'] = fid_score
            print(f"FID: {fid_score:.4f}")
        except Exception as e:
            print(f"FID计算失败: {e}")
            results['fid'] = -1.0
        
        # 2. Inception Score
        print("计算IS...")
        try:
            is_mean, is_std = self.is_calculator.calculate_is(generated_images)
            results['is_mean'] = is_mean
            results['is_std'] = is_std
            print(f"IS: {is_mean:.4f} ± {is_std:.4f}")
        except Exception as e:
            print(f"IS计算失败: {e}")
            results['is_mean'] = -1.0
            results['is_std'] = -1.0
        
        # 3. LPIPS
        print("计算LPIPS...")
        try:
            if len(generated_images) == len(real_images):
                lpips_score = self.lpips_calculator.calculate_lpips(
                    generated_images, real_images
                )
                results['lpips'] = lpips_score
                print(f"LPIPS: {lpips_score:.4f}")
            else:
                print("LPIPS需要相同数量的真实图像和生成图像")
                results['lpips'] = -1.0
        except Exception as e:
            print(f"LPIPS计算失败: {e}")
            results['lpips'] = -1.0
        
        # 4. 医学图像特定指标
        print("计算医学图像指标...")
        try:
            # SSIM和PSNR（如果有配对的真实图像）
            if len(generated_images) == len(real_images):
                ssim_score = self.medical_evaluator.calculate_ssim(
                    generated_images, real_images
                )
                psnr_score = self.medical_evaluator.calculate_psnr(
                    generated_images, real_images
                )
                results['ssim'] = ssim_score
                results['psnr'] = psnr_score
                print(f"SSIM: {ssim_score:.4f}")
                print(f"PSNR: {psnr_score:.4f}")
            
            # 特征多样性
            if panderm_features is not None:
                diversity_metrics = self.medical_evaluator.calculate_feature_diversity(
                    panderm_features
                )
                results['diversity'] = diversity_metrics
                print(f"多样性分数: {diversity_metrics['diversity_score']:.4f}")
            
            # 病变覆盖度
            coverage_metrics = self.medical_evaluator.calculate_lesion_coverage(
                generated_images, real_images
            )
            results['lesion_coverage'] = coverage_metrics
            print(f"颜色特征相似性: {coverage_metrics['color_feature_similarity']:.4f}")
            
        except Exception as e:
            print(f"医学图像指标计算失败: {e}")
        
        # 5. 保存结果
        if save_results:
            self._save_evaluation_results(results, output_dir)
        
        print("评估完成！")
        return results
    
    def _save_evaluation_results(
        self, 
        results: Dict, 
        output_dir: str
    ):
        """保存评估结果"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存为JSON
        import json
        
        # 处理不能序列化的对象
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                                           for k, v in value.items()}
            elif isinstance(value, (int, float, np.number)):
                serializable_results[key] = float(value)
            else:
                serializable_results[key] = str(value)
        
        results_path = os.path.join(output_dir, 'evaluation_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"评估结果已保存到: {results_path}")


if __name__ == "__main__":
    # 测试评估器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建测试数据
    batch_size = 10
    real_images = torch.rand(batch_size, 3, 256, 256)  # [0, 1]范围
    generated_images = torch.rand(batch_size, 3, 256, 256)
    panderm_features = torch.rand(batch_size, 768)
    
    # 创建评估器
    evaluator = ComprehensiveEvaluator(device)
    
    # 运行评估
    results = evaluator.evaluate_generation_quality(
        generated_images=generated_images,
        real_images=real_images,
        panderm_features=panderm_features,
        save_results=True,
        output_dir="./test_evaluation"
    )
    
    print("\n评估结果:")
    for key, value in results.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
    
    print("\n评估器测试完成！")