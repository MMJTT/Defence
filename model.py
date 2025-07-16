"""
模型结构定义模块
构建主干编码器 + 投影头，以及用于剔除B共性的单判别器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.nn.utils import spectral_norm

from config import *


def create_backbone():
    """
    创建预训练的视觉Transformer骨干网络
    
    Returns:
        预训练的backbone模型
    """
    # 创建预训练模型，不包含分类头
    backbone = timm.create_model(BACKBONE, pretrained=True, num_classes=0)
    
    # 冻结前 FREEZE_BLOCKS 个 Transformer block
    for name, param in backbone.named_parameters():
        # 检查参数名是否属于需要冻结的block
        should_freeze = False
        for i in range(FREEZE_BLOCKS):
            if name.startswith(f'blocks.{i}.'):
                should_freeze = True
                break
        
        if should_freeze:
            param.requires_grad = False
            print(f"冻结参数: {name}")
    
    print(f"骨干网络创建完成: {BACKBONE}")
    print(f"冻结前 {FREEZE_BLOCKS} 个 Transformer block")
    
    return backbone


class ProjectionHead(nn.Module):
    """
    投影头网络
    将backbone特征映射到低维表示空间
    架构: 768→512→ReLU→512→128→LayerNorm→ReLU
    """
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, output_dim: int = 128):
        """
        初始化投影头
        
        Args:
            input_dim: 输入特征维度（ViT-Tiny为192，此处假设768用于通用性）
            hidden_dim: 隐藏层维度
            output_dim: 输出特征维度
        """
        super(ProjectionHead, self).__init__()
        
        # 第一层：输入→隐藏
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)
        
        # 第二层：隐藏→输出
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.relu2 = nn.ReLU(inplace=True)
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征张量 [batch_size, input_dim]
            
        Returns:
            投影后的特征张量 [batch_size, output_dim]
        """
        # 第一层变换
        x = self.fc1(x)
        x = self.relu1(x)
        
        # 第二层变换
        x = self.fc2(x)
        x = self.layer_norm(x)
        x = self.relu2(x)
        
        return x


class Discriminator(nn.Module):
    """
    判别器网络
    用于判别特征是否来自B类（保留集）
    架构: 128→64→ReLU + 64→1→Sigmoid，使用SpectralNorm
    """
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64):
        """
        初始化判别器
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
        """
        super(Discriminator, self).__init__()
        
        # 使用 Spectral Normalization 稳定训练
        self.fc1 = spectral_norm(nn.Linear(input_dim, hidden_dim))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = spectral_norm(nn.Linear(hidden_dim, 1))
        self.sigmoid = nn.Sigmoid()
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征张量 [batch_size, input_dim]
            
        Returns:
            判别概率 [batch_size, 1]
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        
        return x


class Encoder(nn.Module):
    """
    编码器网络
    组合backbone和投影头，输出L2归一化的特征表示
    """
    
    def __init__(self):
        """初始化编码器"""
        super(Encoder, self).__init__()
        
        # 创建骨干网络
        self.backbone = create_backbone()
        
        # 获取backbone输出特征维度
        # 对于ViT-Tiny，特征维度通常是192
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            backbone_output = self.backbone(dummy_input)
            feature_dim = backbone_output.shape[1]
        
        # 创建投影头
        self.proj_head = ProjectionHead(
            input_dim=feature_dim, 
            hidden_dim=512, 
            output_dim=128
        )
        
        print(f"编码器初始化完成")
        print(f"骨干网络输出维度: {feature_dim}")
        print(f"投影头输出维度: 128")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像张量 [batch_size, 3, 224, 224]
            
        Returns:
            L2归一化的特征表示 [batch_size, 128]
        """
        # 通过骨干网络提取特征
        feat = self.backbone(x)
        
        # 通过投影头得到低维表示
        z = self.proj_head(feat)
        
        # L2归一化
        z = F.normalize(z, p=2, dim=1)
        
        return z
    
    def get_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取骨干网络特征（用于分析）
        
        Args:
            x: 输入图像张量
            
        Returns:
            骨干网络原始特征
        """
        with torch.no_grad():
            return self.backbone(x)


def create_models():
    """
    创建所有模型组件的便捷函数
    
    Returns:
        Tuple[Encoder, Discriminator]: 编码器和判别器
    """
    # 创建编码器
    encoder = Encoder()
    
    # 创建判别器
    discriminator = Discriminator(input_dim=128, hidden_dim=64)
    
    print("=" * 50)
    print("模型创建完成")
    print(f"编码器参数数量: {sum(p.numel() for p in encoder.parameters() if p.requires_grad):,}")
    print(f"判别器参数数量: {sum(p.numel() for p in discriminator.parameters()):,}")
    print("=" * 50)
    
    return encoder, discriminator


# ======================== 模型测试函数 ========================
def test_models():
    """测试模型前向传播"""
    print("=== 模型测试 ===")
    
    # 创建模型
    encoder, discriminator = create_models()
    
    # 创建测试数据
    batch_size = 4
    test_images = torch.randn(batch_size, 3, 224, 224)
    
    # 测试编码器
    print("\n测试编码器...")
    with torch.no_grad():
        features = encoder(test_images)
        print(f"输入形状: {test_images.shape}")
        print(f"输出特征形状: {features.shape}")
        print(f"特征L2范数: {torch.norm(features, p=2, dim=1)}")
    
    # 测试判别器
    print("\n测试判别器...")
    with torch.no_grad():
        predictions = discriminator(features)
        print(f"判别器输入形状: {features.shape}")
        print(f"判别器输出形状: {predictions.shape}")
        print(f"预测值范围: [{predictions.min().item():.3f}, {predictions.max().item():.3f}]")
    
    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    # 运行模型测试
    test_models()
