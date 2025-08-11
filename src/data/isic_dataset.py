import os
import pandas as pd
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


class ISICDataset(Dataset):
    """ISIC皮肤镜图像数据集加载器"""
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        image_size: int = 224,
        augmentation: bool = True,
        metadata_file: Optional[str] = None
    ):
        """
        Args:
            data_root: 数据集根目录
            split: 数据集分割 ("train", "val", "test")
            image_size: 图像尺寸
            augmentation: 是否启用数据增强
            metadata_file: 元数据CSV文件路径
        """
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        
        # 构建图像路径
        self.image_dir = os.path.join(data_root, "images")
        
        # 加载元数据
        if metadata_file and os.path.exists(metadata_file):
            self.metadata = pd.read_csv(metadata_file)
        else:
            # 如果没有元数据文件，扫描图像目录
            self.metadata = self._scan_images()
        
        # 数据分割
        self.image_paths = self._split_data()
        
        # 数据变换
        self.transform = self._build_transform(augmentation)
        
    def _scan_images(self) -> pd.DataFrame:
        """扫描图像目录构建元数据"""
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend([
                f for f in os.listdir(self.image_dir) 
                if f.lower().endswith(ext)
            ])
        
        # 创建基础元数据
        metadata = pd.DataFrame({
            'image_id': [os.path.splitext(f)[0] for f in image_files],
            'filename': image_files,
            'target': [0] * len(image_files)  # 默认标签
        })
        
        return metadata
    
    def _split_data(self) -> List[str]:
        """数据集分割"""
        image_files = self.metadata['filename'].tolist()
        
        # 简单的8:1:1分割
        train_files, temp_files = train_test_split(
            image_files, test_size=0.2, random_state=42
        )
        val_files, test_files = train_test_split(
            temp_files, test_size=0.5, random_state=42
        )
        
        if self.split == "train":
            return train_files
        elif self.split == "val":
            return val_files
        else:
            return test_files
    
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
            transform_list.insert(-2, transforms.RandomHorizontalFlip(p=0.5))
            transform_list.insert(-2, transforms.RandomVerticalFlip(p=0.5))
            transform_list.insert(-2, transforms.RandomRotation(20))
            transform_list.insert(-2, transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ))
        
        # 标准化 (ImageNet预训练模型标准)
        transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )
        
        return transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个数据样本"""
        filename = self.image_paths[idx]
        image_path = os.path.join(self.image_dir, filename)
        
        # 加载图像
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 返回黑色图像作为fallback
            image = torch.zeros(3, self.image_size, self.image_size)
        
        # 获取元数据信息
        image_id = os.path.splitext(filename)[0]
        metadata_row = self.metadata[
            self.metadata['filename'] == filename
        ]
        
        target = 0  # 默认标签
        if not metadata_row.empty and 'target' in metadata_row.columns:
            target = metadata_row['target'].iloc[0]
        
        return {
            'image': image,
            'image_id': image_id,
            'target': torch.tensor(target, dtype=torch.long),
            'filename': filename
        }


def create_dataloaders(
    data_root: str,
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: int = 224,
    metadata_file: Optional[str] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建训练、验证和测试数据加载器"""
    
    # 创建数据集
    train_dataset = ISICDataset(
        data_root=data_root,
        split="train",
        image_size=image_size,
        augmentation=True,
        metadata_file=metadata_file
    )
    
    val_dataset = ISICDataset(
        data_root=data_root,
        split="val", 
        image_size=image_size,
        augmentation=False,
        metadata_file=metadata_file
    )
    
    test_dataset = ISICDataset(
        data_root=data_root,
        split="test",
        image_size=image_size,
        augmentation=False,
        metadata_file=metadata_file
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
    data_root = "./data/ISIC"
    
    if os.path.exists(data_root):
        train_loader, val_loader, test_loader = create_dataloaders(
            data_root=data_root,
            batch_size=4,
            num_workers=0
        )
        
        print(f"训练集样本数: {len(train_loader.dataset)}")
        print(f"验证集样本数: {len(val_loader.dataset)}")  
        print(f"测试集样本数: {len(test_loader.dataset)}")
        
        # 测试加载一个batch
        for batch in train_loader:
            print(f"图像张量形状: {batch['image'].shape}")
            print(f"图像ID示例: {batch['image_id'][:2]}")
            print(f"标签: {batch['target']}")
            break
    else:
        print(f"数据目录不存在: {data_root}")
        print("请将ISIC数据集放置在正确的目录下")