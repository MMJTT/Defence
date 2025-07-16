"""
损失函数模块
封装 InfoNCE、NCL、对抗、差异向量等所有损失，并提供 total_loss 接口
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any

from config import *


def info_nce(z1: torch.Tensor, z2: torch.Tensor, z_neg: torch.Tensor, 
            tau: float = TAU) -> torch.Tensor:
    """
    InfoNCE损失 - 标准对比学习损失
    
    Args:
        z1: 正对样本1 [batch_size, feature_dim]
        z2: 正对样本2 [batch_size, feature_dim]  
        z_neg: 负样本 [num_neg, feature_dim]
        tau: 温度参数
        
    Returns:
        InfoNCE损失值
    """
    if z1.numel() == 0 or z2.numel() == 0:
        return torch.tensor(0.0, device=z1.device if z1.numel() > 0 else z2.device)
    
    # 确保z1和z2有相同的batch size
    batch_size = min(z1.shape[0], z2.shape[0])
    z1 = z1[:batch_size]
    z2 = z2[:batch_size]
    
    # 计算正对相似度
    pos_sim = torch.cosine_similarity(z1, z2, dim=-1)  # [batch_size]
    pos_exp = torch.exp(pos_sim / tau)  # [batch_size]
    
    # 计算z1与所有负样本的相似度
    if z_neg.numel() > 0:
        # z1与z_neg的相似度矩阵
        neg_sim = torch.mm(F.normalize(z1, p=2, dim=1), 
                          F.normalize(z_neg, p=2, dim=1).t())  # [batch_size, num_neg]
        neg_exp = torch.exp(neg_sim / tau)  # [batch_size, num_neg]
        neg_sum = torch.sum(neg_exp, dim=1)  # [batch_size]
    else:
        neg_sum = torch.zeros_like(pos_exp)
    
    # InfoNCE损失: -mean(log(pos / (pos + neg_sum)))
    denominator = pos_exp + neg_sum
    loss = -torch.log(pos_exp / (denominator + 1e-8))
    
    return loss.mean()


def ncl_loss(zA: torch.Tensor, lambda_ncl: float) -> torch.Tensor:
    """
    NCL损失 - A类特征的正则化损失
    
    Args:
        zA: A类特征 [num_A, feature_dim]
        lambda_ncl: NCL损失权重
        
    Returns:
        NCL损失值
    """
    if zA.numel() == 0:
        return torch.tensor(0.0, device=zA.device if zA.numel() > 0 else torch.device('cpu'))
    
    # 计算 relu(zA).abs().mean() * lambda_ncl
    loss = torch.relu(zA).abs().mean() * lambda_ncl
    
    return loss


def adv_loss(D: nn.Module, zA: torch.Tensor, zB: torch.Tensor, 
            lambda_adv: float) -> torch.Tensor:
    """
    对抗损失 - 编码器欺骗判别器
    
    Args:
        D: 判别器模型
        zA: A类特征 [num_A, feature_dim]
        zB: B类特征 [num_B, feature_dim]
        lambda_adv: 对抗损失权重
        
    Returns:
        对抗损失值
    """
    total_loss = torch.tensor(0.0, device=next(D.parameters()).device)
    count = 0
    
    # BCE(D(zA), 1) - 希望判别器认为A类是B类
    if zA.numel() > 0:
        pred_zA = D(zA).squeeze()
        target_A = torch.ones_like(pred_zA)
        loss_A = F.binary_cross_entropy(pred_zA, target_A)
        total_loss += loss_A
        count += 1
    
    # BCE(D(zB), 0) - 希望判别器认为B类不是B类  
    if zB.numel() > 0:
        pred_zB = D(zB).squeeze()
        target_B = torch.zeros_like(pred_zB)
        loss_B = F.binary_cross_entropy(pred_zB, target_B)
        total_loss += loss_B
        count += 1
    
    # 返回 -lambda_adv * 平均损失
    avg_loss = total_loss / max(count, 1)
    return -lambda_adv * avg_loss


def diff_loss(zA: torch.Tensor, muA: torch.Tensor, muB: torch.Tensor, 
             lambda_diff: float) -> torch.Tensor:
    """
    差异损失 - 促进A类特征在muA-muB方向上的投影
    
    Args:
        zA: A类特征 [num_A, feature_dim]
        muA: A类均值向量 [feature_dim]
        muB: B类均值向量 [feature_dim]
        lambda_diff: 差异损失权重
        
    Returns:
        差异损失值
    """
    if zA.numel() == 0:
        return torch.tensor(0.0, device=muA.device)
    
    # 计算差异向量并归一化
    diff_vec = muA - muB  # [feature_dim]
    diff_norm = torch.norm(diff_vec, p=2)
    
    if diff_norm < 1e-8:
        return torch.tensor(0.0, device=muA.device)
    
    v = diff_vec / diff_norm  # 归一化差异向量 [feature_dim]
    
    # 计算zA在v方向上的投影
    projections = torch.mm(zA, v.unsqueeze(1)).squeeze()  # [num_A]
    
    # 返回 -lambda_diff * mean(projections^2)
    loss = -lambda_diff * torch.mean(projections ** 2)
    
    return loss


def discriminator_loss(discriminator: nn.Module, zA: torch.Tensor, 
                      zB: torch.Tensor) -> torch.Tensor:
    """
    判别器损失 - 训练判别器区分A类和B类
    
    Args:
        discriminator: 判别器模型
        zA: A类特征 [num_A, feature_dim]
        zB: B类特征 [num_B, feature_dim]
        
    Returns:
        判别器损失值
    """
    total_loss = torch.tensor(0.0, device=next(discriminator.parameters()).device)
    count = 0
    
    # A类应该被预测为0（不是B类）
    if zA.numel() > 0:
        pred_A = discriminator(zA).squeeze()
        target_A = torch.zeros_like(pred_A)
        loss_A = F.binary_cross_entropy(pred_A, target_A)
        total_loss += loss_A
        count += 1
    
    # B类应该被预测为1（是B类）
    if zB.numel() > 0:
        pred_B = discriminator(zB).squeeze()
        target_B = torch.ones_like(pred_B)
        loss_B = F.binary_cross_entropy(pred_B, target_B)
        total_loss += loss_B
        count += 1
    
    return total_loss / max(count, 1)


def total_loss(zA: torch.Tensor, zB: torch.Tensor, z_neg: torch.Tensor,
               D: nn.Module, muA: torch.Tensor, muB: torch.Tensor,
               cfg: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    计算总损失 - 按配置权重加权所有损失项
    
    Args:
        zA: A类特征 [num_A, feature_dim]
        zB: B类特征 [num_B, feature_dim]
        z_neg: 负样本特征 [num_neg, feature_dim]
        D: 判别器模型
        muA: A类均值向量 [feature_dim]
        muB: B类均值向量 [feature_dim]
        cfg: 配置模块
        
    Returns:
        Tuple[总损失, 各项损失字典]
    """
    # 为InfoNCE准备正对样本
    if zA.numel() >= 2:
        # 从zA中选择两个样本作为正对
        indices = torch.randperm(zA.shape[0])[:2]
        z1, z2 = zA[indices[0]:indices[0]+1], zA[indices[1]:indices[1]+1]
    elif zA.numel() == 1:
        # 如果只有一个A样本，复制它
        z1, z2 = zA, zA.clone()
    else:
        # 如果没有A样本，创建零向量
        device = muA.device
        z1 = torch.zeros(1, muA.shape[0], device=device)
        z2 = torch.zeros(1, muA.shape[0], device=device)
    
    # 计算各项损失
    loss_info = info_nce(z1, z2, z_neg, cfg.TAU)
    loss_ncl = ncl_loss(zA, cfg.LAMBDA_NCL)
    loss_adv = adv_loss(D, zA, zB, cfg.LAMBDA_ADV)
    loss_diff = diff_loss(zA, muA, muB, cfg.LAMBDA_DIFF)
    
    # 加权求和
    total = (cfg.LAMBDA_INFO * loss_info +
             loss_ncl +  # 已经包含lambda_ncl权重
             loss_adv +  # 已经包含lambda_adv权重
             loss_diff)  # 已经包含lambda_diff权重
    
    # 损失字典
    loss_dict = {
        'loss_info': loss_info.item(),
        'loss_ncl': loss_ncl.item(),
        'loss_adv': loss_adv.item(),
        'loss_diff': loss_diff.item()
    }
    
    return total, loss_dict


# ======================== 测试函数 ========================
if __name__ == "__main__":
    """测试损失函数"""
    print("=== 损失函数测试 ===")
    
    # 创建测试数据
    batch_size = 8
    feature_dim = 128
    
    zA = torch.randn(batch_size, feature_dim)
    zB = torch.randn(batch_size, feature_dim)
    zO = torch.randn(batch_size, feature_dim)
    z_neg = torch.cat([zB, zO], dim=0)
    
    muA = torch.randn(feature_dim)
    muB = torch.randn(feature_dim)
    
    # 创建简单判别器
    D = nn.Sequential(
        nn.Linear(feature_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid()
    )
    
    # 测试各项损失
    print("测试InfoNCE损失...")
    z1, z2 = zA[:2], zA[2:4]
    loss_info = info_nce(z1, z2, z_neg)
    print(f"InfoNCE损失: {loss_info.item():.4f}")
    
    print("测试NCL损失...")
    loss_ncl = ncl_loss(zA, 1.0)
    print(f"NCL损失: {loss_ncl.item():.4f}")
    
    print("测试对抗损失...")
    loss_adv = adv_loss(D, zA, zB, 0.5)
    print(f"对抗损失: {loss_adv.item():.4f}")
    
    print("测试差异损失...")
    loss_diff = diff_loss(zA, muA, muB, 0.3)
    print(f"差异损失: {loss_diff.item():.4f}")
    
    print("测试判别器损失...")
    loss_d = discriminator_loss(D, zA, zB)
    print(f"判别器损失: {loss_d.item():.4f}")
    
    print("测试总损失...")
    import config as cfg
    total, loss_dict = total_loss(zA, zB, z_neg, D, muA, muB, cfg)
    print(f"总损失: {total.item():.4f}")
    print(f"损失分解: {loss_dict}")
    
    print("=== 测试完成 ===")
