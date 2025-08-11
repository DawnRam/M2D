import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from typing import Tuple, List, Optional
import torch
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler


class MedicalImagePreprocessor:
    """医学图像预处理工具"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        
    def remove_hair_artifacts(self, image: np.ndarray) -> np.ndarray:
        """去除皮肤镜图像中的毛发伪影"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # 使用黑帽变换检测毛发
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        # 创建毛发掩码
        ret, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        
        # 使用inpainting去除毛发
        if len(image.shape) == 3:
            result = cv2.inpaint(image, hair_mask, 3, cv2.INPAINT_TELEA)
        else:
            result = cv2.inpaint(image, hair_mask, 3, cv2.INPAINT_TELEA)
            
        return result
    
    def enhance_contrast(self, image: np.ndarray, factor: float = 1.2) -> np.ndarray:
        """增强图像对比度"""
        if len(image.shape) == 3:
            # RGB图像
            pil_image = Image.fromarray(image.astype('uint8'))
            enhancer = ImageEnhance.Contrast(pil_image)
            enhanced = enhancer.enhance(factor)
            return np.array(enhanced)
        else:
            # 灰度图像
            return cv2.convertScaleAbs(image, alpha=factor, beta=0)
    
    def crop_lesion_roi(self, image: np.ndarray, margin_ratio: float = 0.1) -> np.ndarray:
        """裁剪病变感兴趣区域"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # 使用Otsu阈值分割
        ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 找到最大轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # 添加边距
            margin_x = int(w * margin_ratio)
            margin_y = int(h * margin_ratio)
            
            x = max(0, x - margin_x)
            y = max(0, y - margin_y)
            w = min(image.shape[1] - x, w + 2 * margin_x)
            h = min(image.shape[0] - y, h + 2 * margin_y)
            
            # 裁剪
            if len(image.shape) == 3:
                cropped = image[y:y+h, x:x+w, :]
            else:
                cropped = image[y:y+h, x:x+w]
                
            return cropped
        
        return image
    
    def normalize_illumination(self, image: np.ndarray) -> np.ndarray:
        """标准化光照条件"""
        if len(image.shape) == 3:
            # 转换到LAB色彩空间
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0]
            
            # 应用CLAHE到L通道
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)
            
            lab[:, :, 0] = l_channel
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            return result
        else:
            # 灰度图像直接应用CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
    
    def preprocess_pipeline(
        self, 
        image: np.ndarray,
        remove_hair: bool = True,
        enhance_contrast: bool = True,
        crop_roi: bool = False,
        normalize_illumination: bool = True
    ) -> np.ndarray:
        """完整的预处理管道"""
        result = image.copy()
        
        # 去除毛发伪影
        if remove_hair:
            result = self.remove_hair_artifacts(result)
        
        # 增强对比度
        if enhance_contrast:
            result = self.enhance_contrast(result)
        
        # 裁剪ROI
        if crop_roi:
            result = self.crop_lesion_roi(result)
        
        # 标准化光照
        if normalize_illumination:
            result = self.normalize_illumination(result)
        
        # 调整尺寸
        if len(result.shape) == 3:
            result = cv2.resize(result, self.target_size, interpolation=cv2.INTER_CUBIC)
        else:
            result = cv2.resize(result, self.target_size, interpolation=cv2.INTER_CUBIC)
            
        return result


class DataAugmentation:
    """医学图像数据增强"""
    
    def __init__(self, p: float = 0.5):
        self.p = p
        
    def get_training_transforms(self) -> transforms.Compose:
        """获取训练时的数据增强变换"""
        return transforms.Compose([
            transforms.ToPILImage(),
            
            # 几何变换
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=20, expand=False),
            
            # 颜色变换
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2, 
                saturation=0.2,
                hue=0.1
            ),
            
            # 仿射变换
            transforms.RandomAffine(
                degrees=10,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=5
            ),
            
            # 弹性变换（可选）
            # transforms.ElasticTransform(alpha=50.0, sigma=5.0),
            
            transforms.ToTensor(),
        ])
    
    def get_validation_transforms(self) -> transforms.Compose:
        """获取验证时的变换（无增强）"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])


def batch_preprocess_images(
    input_dir: str,
    output_dir: str,
    processor: Optional[MedicalImagePreprocessor] = None
) -> None:
    """批量预处理图像"""
    if processor is None:
        processor = MedicalImagePreprocessor()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 支持的图像格式
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    image_files = []
    for ext in supported_formats:
        image_files.extend([
            f for f in os.listdir(input_dir)
            if f.lower().endswith(ext)
        ])
    
    print(f"找到 {len(image_files)} 张图像待处理...")
    
    for i, filename in enumerate(image_files):
        try:
            # 读取图像
            input_path = os.path.join(input_dir, filename)
            image = cv2.imread(input_path)
            
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 预处理
                processed = processor.preprocess_pipeline(image)
                processed = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
                
                # 保存
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, processed)
                
                if (i + 1) % 100 == 0:
                    print(f"已处理 {i + 1}/{len(image_files)} 张图像")
                    
        except Exception as e:
            print(f"处理图像 {filename} 时出错: {e}")
    
    print("批量预处理完成！")


if __name__ == "__main__":
    # 测试预处理功能
    processor = MedicalImagePreprocessor()
    
    # 创建一个测试图像
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # 测试预处理管道
    processed = processor.preprocess_pipeline(test_image)
    
    print(f"原始图像形状: {test_image.shape}")
    print(f"处理后图像形状: {processed.shape}")
    print("预处理测试完成！")