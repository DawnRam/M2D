"""
ISIC多类别皮肤病数据集加载器
支持7个类别的ISIC数据集
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import json


class ISICMultiClassDataset(Dataset):
    """ISIC多类别皮肤镜图像数据集加载器"""
    
    # ISIC数据集的7个类别
    CLASSES = {
        'AKIEC': 0,  # 光化性角化病 (Actinic keratoses)
        'BCC': 1,    # 基底细胞癌 (Basal cell carcinoma)
        'BKL': 2,    # 良性角化病 (Benign keratosis-like lesions)
        'DF': 3,     # 皮肤纤维瘤 (Dermatofibroma)
        'MEL': 4,    # 黑色素瘤 (Melanoma)
        'NV': 5,     # 色素痣 (Melanocytic nevi)
        'VASC': 6    # 血管病变 (Vascular lesions)
    }
    
    # 类别中文名称
    CLASS_NAMES_CN = {
        'AKIEC': '光化性角化病',
        'BCC': '基底细胞癌',
        'BKL': '良性角化病',
        'DF': '皮肤纤维瘤',
        'MEL': '黑色素瘤',
        'NV': '色素痣',
        'VASC': '血管病变'
    }
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        image_size: int = 256,
        augmentation: bool = True,
        balance_classes: bool = False,
        cache_path: Optional[str] = None
    ):
        """
        Args:
            data_root: 数据集根目录，包含7个类别子目录
            split: 数据集分割 ("train", "val", "test")
            image_size: 图像尺寸
            augmentation: 是否启用数据增强
            balance_classes: 是否进行类别平衡采样
            cache_path: 缓存文件路径（用于加速数据加载）
        """
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        self.augmentation = augmentation
        self.balance_classes = balance_classes
        
        # 检查数据目录
        if not os.path.exists(data_root):
            raise ValueError(f"数据目录不存在: {data_root}")
        
        # 扫描并构建数据集
        self.samples = self._scan_dataset(cache_path)
        
        # 数据分割
        self.samples = self._split_data()
        
        # 类别平衡
        if balance_classes and split == "train":
            self.samples = self._balance_samples()
        
        # 数据变换
        self.transform = self._build_transform(augmentation)
        
        # 打印数据集信息
        self._print_dataset_info()
    
    def _scan_dataset(self, cache_path: Optional[str] = None) -> List[Dict]:
        """扫描数据集目录，构建样本列表"""
        
        # 尝试从缓存加载
        if cache_path and os.path.exists(cache_path):
            print(f"从缓存加载数据集信息: {cache_path}")
            with open(cache_path, 'r') as f:
                return json.load(f)
        
        samples = []
        
        # 遍历每个类别目录
        for class_name, class_idx in self.CLASSES.items():
            class_dir = os.path.join(self.data_root, class_name)
            
            if not os.path.exists(class_dir):
                print(f"⚠ 警告：类别目录不存在: {class_dir}")
                continue
            
            # 扫描该类别下的所有图像
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            for img_file in image_files:
                samples.append({
                    'path': os.path.join(class_dir, img_file),
                    'filename': img_file,
                    'class_name': class_name,
                    'class_idx': class_idx,
                    'class_name_cn': self.CLASS_NAMES_CN[class_name]
                })
            
            print(f"✓ 扫描 {class_name} ({self.CLASS_NAMES_CN[class_name]}): {len(image_files)} 张图像")
        
        # 保存到缓存
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(samples, f)
            print(f"✓ 数据集信息已缓存到: {cache_path}")
        
        return samples
    
    def _split_data(self) -> List[Dict]:
        """数据集分割"""
        # 按类别分组
        class_samples = {i: [] for i in range(len(self.CLASSES))}
        for sample in self.samples:
            class_samples[sample['class_idx']].append(sample)
        
        # 对每个类别进行分割
        train_samples = []
        val_samples = []
        test_samples = []
        
        for class_idx, samples in class_samples.items():
            if len(samples) == 0:
                continue
                
            # 8:1:1 分割
            n_samples = len(samples)
            n_train = int(n_samples * 0.8)
            n_val = int(n_samples * 0.1)
            
            # 随机打乱
            np.random.seed(42)
            np.random.shuffle(samples)
            
            train_samples.extend(samples[:n_train])
            val_samples.extend(samples[n_train:n_train + n_val])
            test_samples.extend(samples[n_train + n_val:])
        
        # 返回对应分割的数据
        if self.split == "train":
            return train_samples
        elif self.split == "val":
            return val_samples
        else:
            return test_samples
    
    def _balance_samples(self) -> List[Dict]:
        """类别平衡采样（通过过采样）"""
        # 统计每个类别的样本数
        class_counts = {}
        for sample in self.samples:
            class_idx = sample['class_idx']
            class_counts[class_idx] = class_counts.get(class_idx, 0) + 1
        
        # 找到最多的类别样本数
        max_count = max(class_counts.values())
        
        # 对每个类别进行过采样
        balanced_samples = []
        for class_idx in range(len(self.CLASSES)):
            class_samples = [s for s in self.samples if s['class_idx'] == class_idx]
            
            if len(class_samples) == 0:
                continue
            
            # 过采样到最大数量
            n_repeat = max_count // len(class_samples)
            n_extra = max_count % len(class_samples)
            
            balanced_samples.extend(class_samples * n_repeat)
            if n_extra > 0:
                balanced_samples.extend(np.random.choice(class_samples, n_extra).tolist())
        
        print(f"✓ 类别平衡完成：每个类别约 {max_count} 个样本")
        return balanced_samples
    
    def _build_transform(self, augmentation: bool) -> transforms.Compose:
        """构建数据变换管道"""
        transform_list = []
        
        # 基础变换
        transform_list.extend([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])
        
        # 数据增强（仅训练集）
        if augmentation and self.split == "train":
            # 医学图像专用增强
            transform_list.insert(-1, transforms.RandomHorizontalFlip(p=0.5))
            transform_list.insert(-1, transforms.RandomVerticalFlip(p=0.5))
            transform_list.insert(-1, transforms.RandomRotation(30))
            transform_list.insert(-1, transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ))
            transform_list.insert(-1, transforms.ColorJitter(
                brightness=0.3, 
                contrast=0.3, 
                saturation=0.3, 
                hue=0.1
            ))
            # 随机擦除
            transform_list.append(transforms.RandomErasing(p=0.2))
        
        # 标准化 (ImageNet预训练模型标准)
        transform_list.insert(-1, transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ))
        
        return transforms.Compose(transform_list)
    
    def _print_dataset_info(self):
        """打印数据集统计信息"""
        class_counts = {}
        for sample in self.samples:
            class_name = sample['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"\n{'='*50}")
        print(f"ISIC数据集 - {self.split}集")
        print(f"{'='*50}")
        print(f"总样本数: {len(self.samples)}")
        print(f"类别分布:")
        for class_name in self.CLASSES.keys():
            count = class_counts.get(class_name, 0)
            percentage = count / len(self.samples) * 100 if len(self.samples) > 0 else 0
            print(f"  {class_name:6} ({self.CLASS_NAMES_CN[class_name]:8}): {count:5} ({percentage:5.1f}%)")
        print(f"{'='*50}\n")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个数据样本"""
        sample = self.samples[idx]
        
        # 加载图像
        try:
            image = Image.open(sample['path']).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {sample['path']}: {e}")
            # 返回黑色图像作为fallback
            image = torch.zeros(3, self.image_size, self.image_size)
        
        return {
            'image': image,
            'label': torch.tensor(sample['class_idx'], dtype=torch.long),
            'class_name': sample['class_name'],
            'class_name_cn': sample['class_name_cn'],
            'filename': sample['filename']
        }
    
    def get_class_weights(self) -> torch.Tensor:
        """计算类别权重（用于处理类别不平衡）"""
        class_counts = torch.zeros(len(self.CLASSES))
        for sample in self.samples:
            class_counts[sample['class_idx']] += 1
        
        # 使用逆频率作为权重
        total_samples = len(self.samples)
        class_weights = total_samples / (len(self.CLASSES) * class_counts)
        
        return class_weights


def create_isic_dataloaders(
    data_root: str,
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: int = 256,
    augmentation: bool = True,
    balance_classes: bool = False,
    cache_dir: Optional[str] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建ISIC数据加载器"""
    
    # 缓存路径
    if cache_dir is not None:
        cache_path = os.path.join(cache_dir, "isic_dataset_cache.json")
        os.makedirs(cache_dir, exist_ok=True)
    else:
        cache_path = None
    
    # 创建数据集
    train_dataset = ISICMultiClassDataset(
        data_root=data_root,
        split="train",
        image_size=image_size,
        augmentation=augmentation,
        balance_classes=balance_classes,
        cache_path=cache_path
    )
    
    val_dataset = ISICMultiClassDataset(
        data_root=data_root,
        split="val",
        image_size=image_size,
        augmentation=False,
        balance_classes=False,
        cache_path=cache_path
    )
    
    test_dataset = ISICMultiClassDataset(
        data_root=data_root,
        split="test",
        image_size=image_size,
        augmentation=False,
        balance_classes=False,
        cache_path=cache_path
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 测试数据加载器
    data_root = "/nfs/scratch/eechengyang/Data/ISIC"
    
    if os.path.exists(data_root):
        train_loader, val_loader, test_loader = create_isic_dataloaders(
            data_root=data_root,
            batch_size=4,
            num_workers=0,
            balance_classes=True
        )
        
        print(f"\n数据加载器创建成功！")
        print(f"训练批次数: {len(train_loader)}")
        print(f"验证批次数: {len(val_loader)}")
        print(f"测试批次数: {len(test_loader)}")
        
        # 测试加载一个batch
        print("\n测试加载一个批次...")
        for batch in train_loader:
            print(f"图像张量形状: {batch['image'].shape}")
            print(f"标签张量形状: {batch['label'].shape}")
            print(f"类别名称: {batch['class_name']}")
            print(f"中文名称: {batch['class_name_cn']}")
            break
    else:
        print(f"数据目录不存在: {data_root}")