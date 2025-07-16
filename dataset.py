"""
数据加载与批次采样模块
实现 CIFAR-10 子集划分和自定义 BatchSampler，确保每个 batch 固定包含 A/B/O 指定数量
"""

import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, BatchSampler
from torchvision import datasets, transforms
from typing import List, Tuple, Any
import json

from config import *


class CIFAR10Subset(Dataset):
    """
    CIFAR-10 子集数据集类
    根据遗忘索引和标签将数据分为 A（遗忘集）、B（保留集-鸟类）、O（其他）三类
    """
    
    def __init__(self, root: str, train: bool = True, forget_indices: List[int] = None, 
                 transform=None):
        """
        初始化 CIFAR-10 子集
        
        Args:
            root: 数据集根目录
            train: 是否为训练集
            forget_indices: 遗忘集索引列表（A类）
            transform: 数据变换
        """
        self.cifar_dataset = datasets.CIFAR10(root=root, train=train, download=True)
        self.forget_indices = set(forget_indices) if forget_indices else set()
        self.transform = transform or self._default_transform()
        
        # CIFAR-10 类别名称（索引2为bird）
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck']
    
    def _default_transform(self):
        """默认数据变换"""
        return transforms.Compose([
            transforms.Resize(224),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.cifar_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        获取数据项
        
        Args:
            idx: 数据索引
            
        Returns:
            Tuple[image, tag]: 图像张量和标签字符串
        """
        image, label = self.cifar_dataset[idx]
        
        # 应用数据变换
        if self.transform:
            image = self.transform(image)
        
        # 确定标签类型
        if idx in self.forget_indices:
            tag = 'A'  # 遗忘集
        elif label == 2 and idx not in self.forget_indices:  # bird类且不在遗忘集
            tag = 'B'  # 保留集
        else:
            tag = 'O'  # 其他
            
        return image, tag


class BalancedBatchSampler(BatchSampler):
    """
    平衡批次采样器
    确保每个 batch 包含固定数量的 A/B/O 类型样本
    """
    
    def __init__(self, forget_indices: List[int], total_size: int, 
                 batch_size: int, ratios: Tuple[int, int, int]):
        """
        初始化平衡采样器
        
        Args:
            forget_indices: 遗忘集索引列表
            total_size: 数据集总大小
            batch_size: 批次大小
            ratios: (A类数量, B类数量, O类数量) 元组
        """
        self.forget_indices = set(forget_indices)
        self.total_size = total_size
        self.batch_size = batch_size
        self.batch_a, self.batch_b, self.batch_o = ratios
        
        # 验证批次配置
        assert self.batch_a + self.batch_b + self.batch_o == batch_size, \
            f"批次配置错误: {self.batch_a} + {self.batch_b} + {self.batch_o} != {batch_size}"
        
        # 预先分类所有索引
        self._categorize_indices()
    
    def _categorize_indices(self):
        """将所有索引按 A/B/O 分类"""
        self.indices_a = []  # 遗忘集索引
        self.indices_b = []  # 保留集索引（鸟类）
        self.indices_o = []  # 其他索引
        
        # 这里需要加载CIFAR-10来获取标签信息
        cifar_dataset = datasets.CIFAR10(root=CIFAR_DATA_PATH.replace('.tar.gz', ''), 
                                        train=True, download=False)
        
        # 只处理实际存在的索引范围
        max_idx = min(self.total_size, len(cifar_dataset))
        
        for idx in range(max_idx):
            _, label = cifar_dataset[idx]
            
            if idx in self.forget_indices:
                self.indices_a.append(idx)
            elif label == 2 and idx not in self.forget_indices:  # bird类且不在遗忘集
                self.indices_b.append(idx)
            else:
                self.indices_o.append(idx)
        
        print(f"数据分类统计: A类={len(self.indices_a)}, B类={len(self.indices_b)}, O类={len(self.indices_o)}")
    
    def __iter__(self):
        """生成平衡的批次索引"""
        num_batches = len(self)
        
        for _ in range(num_batches):
            batch_indices = []
            
            # 从 A 类中随机采样
            if len(self.indices_a) >= self.batch_a:
                batch_indices.extend(random.sample(self.indices_a, self.batch_a))
            elif len(self.indices_a) > 0:
                # 如果 A 类样本不足，进行重复采样
                batch_indices.extend(random.choices(self.indices_a, k=self.batch_a))
            else:
                # 如果没有A类样本，用B类替代
                batch_indices.extend(random.choices(self.indices_b[:self.batch_a] if len(self.indices_b) >= self.batch_a else self.indices_b, k=self.batch_a))
            
            # 从 B 类中随机采样
            if len(self.indices_b) >= self.batch_b:
                batch_indices.extend(random.sample(self.indices_b, self.batch_b))
            elif len(self.indices_b) > 0:
                # 如果 B 类样本不足，进行重复采样
                batch_indices.extend(random.choices(self.indices_b, k=self.batch_b))
            else:
                # 如果没有B类样本，用O类替代
                batch_indices.extend(random.choices(self.indices_o[:self.batch_b] if len(self.indices_o) >= self.batch_b else self.indices_o, k=self.batch_b))
            
            # 从 O 类中随机采样
            if len(self.indices_o) >= self.batch_o:
                batch_indices.extend(random.sample(self.indices_o, self.batch_o))
            elif len(self.indices_o) > 0:
                # 如果 O 类样本不足，进行重复采样
                batch_indices.extend(random.choices(self.indices_o, k=self.batch_o))
            else:
                # 如果没有O类样本，用B类替代
                batch_indices.extend(random.choices(self.indices_b[:self.batch_o] if len(self.indices_b) >= self.batch_o else self.indices_b, k=self.batch_o))
            
            # 确保所有索引都在有效范围内
            batch_indices = [idx for idx in batch_indices if idx < self.total_size]
            
            # 如果批次不够大，补充索引
            while len(batch_indices) < self.batch_size:
                available_indices = self.indices_a + self.indices_b + self.indices_o
                if available_indices:
                    batch_indices.append(random.choice(available_indices))
                else:
                    break
            
            # 打乱批次内索引顺序
            random.shuffle(batch_indices)
            yield batch_indices[:self.batch_size]  # 确保批次大小正确
    
    def __len__(self) -> int:
        """返回批次数量"""
        return self.total_size // self.batch_size


def load_forget_indices(file_path: str) -> List[int]:
    """
    从文件加载遗忘集索引
    
    Args:
        file_path: 遗忘集索引文件路径
        
    Returns:
        遗忘集索引列表
    """
    try:
        with open(file_path, 'r') as f:
            forget_indices = json.load(f)
        return forget_indices
    except FileNotFoundError:
        print(f"警告: 未找到遗忘集索引文件 {file_path}，将生成默认索引")
        # 生成默认的遗忘集索引（前50个）
        return list(range(A_SIZE))


def create_dataloader(train: bool = True, num_workers: int = 4) -> DataLoader:
    """
    创建数据加载器的便捷函数
    
    Args:
        train: 是否为训练集
        num_workers: 工作进程数
        
    Returns:
        配置好的 DataLoader
    """
    # 加载遗忘集索引
    forget_indices = load_forget_indices(FORGET_INDICES_FILE)
    
    # 创建数据集
    dataset = CIFAR10Subset(
        root=CIFAR_DATA_PATH.replace('.tar.gz', ''),
        train=train,
        forget_indices=forget_indices
    )
    
    # 创建平衡采样器 - 确保使用正确的数据集大小
    batch_size = BATCH_A + BATCH_B + BATCH_O
    sampler = BalancedBatchSampler(
        forget_indices=forget_indices,
        total_size=len(dataset),
        batch_size=batch_size,
        ratios=(BATCH_A, BATCH_B, BATCH_O)
    )
    
    # 创建数据加载器 - 减少worker数量避免索引问题
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=min(num_workers, 2),  # 减少worker数量
        pin_memory=True
    )
    
    return dataloader


# ======================== 使用示例 ========================
if __name__ == "__main__":
    """
    使用示例和测试代码
    """
    print("=== CIFAR-10 平衡数据加载器测试 ===")
    
    # 创建训练数据加载器
    train_loader = create_dataloader(train=True, num_workers=2)
    
    print(f"批次大小: {BATCH_A + BATCH_B + BATCH_O}")
    print(f"A类:{BATCH_A}, B类:{BATCH_B}, O类:{BATCH_O}")
    print(f"总批次数: {len(train_loader)}")
    
    # 测试一个批次
    for batch_idx, (images, tags) in enumerate(train_loader):
        print(f"\n批次 {batch_idx + 1}:")
        print(f"  图像形状: {images.shape}")
        
        # 统计标签分布
        tag_counts = {'A': 0, 'B': 0, 'O': 0}
        for tag in tags:
            tag_counts[tag] += 1
        
        print(f"  标签分布: A={tag_counts['A']}, B={tag_counts['B']}, O={tag_counts['O']}")
        
        # 只测试前3个批次
        if batch_idx >= 2:
            break
    
    print("\n=== 测试完成 ===")
