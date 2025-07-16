"""
程序入口与训练主流程
组装 config、dataset、model、losses、utils，执行完整训练实验并输出可视化结果
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import json

# 导入所有模块
import config
from dataset import CIFAR10Subset, BalancedBatchSampler, create_dataloader, load_forget_indices
from model import Encoder, Discriminator, create_models
import losses
from utils import (
    update_ema, split_by_tag, visualize_tsne,
    ExperimentLogger, save_checkpoint, load_checkpoint
)


class FeatureExtractionTrainer:
    """
    A类特征提取训练器
    实现完整的对抗训练流程
    """
    
    def __init__(self, args):
        """
        初始化训练器
        
        Args:
            args: 命令行参数
        """
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 设置随机种子
        torch.manual_seed(42)
        np.random.seed(42)
        
        # 初始化实验记录器
        self.logger = ExperimentLogger("feature_extraction_experiment")
        
        # 创建数据加载器
        self._create_dataloaders()
        
        # 创建模型
        self._create_models()
        
        # 创建优化器和调度器
        self._create_optimizers()
        
        # 初始化EMA参数
        self._initialize_ema()
        
        # 训练状态
        self.best_loss = float('inf')
        self.start_epoch = 0
        
        # 如果有检查点，加载它
        if args.resume:
            self._load_checkpoint(args.resume)
    
    def _create_dataloaders(self):
        """创建数据加载器"""
        print("创建数据加载器...")
        
        # 训练数据加载器
        self.train_loader = create_dataloader(train=True, num_workers=4)
        
        # 验证数据加载器（用于可视化）
        self.eval_loader = create_dataloader(train=False, num_workers=2)
        
        print(f"训练批次数: {len(self.train_loader)}")
        print(f"验证批次数: {len(self.eval_loader)}")
    
    def _create_models(self):
        """创建模型"""
        print("创建模型...")
        
        # 创建编码器和判别器
        self.encoder, self.discriminator = create_models()
        
        # 移动到设备
        self.encoder = self.encoder.to(self.device)
        self.discriminator = self.discriminator.to(self.device)
        
        print("模型创建完成")
    
    def _create_optimizers(self):
        """创建优化器和学习率调度器"""
        print("创建优化器...")
        
        # 编码器优化器
        self.opt_encoder = optim.AdamW(
            self.encoder.parameters(),
            lr=config.LR_ENCODER,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # 判别器优化器
        self.opt_discriminator = optim.AdamW(
            self.discriminator.parameters(),
            lr=config.LR_DISCRIM,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # 学习率调度器
        self.scheduler = CosineAnnealingLR(
            self.opt_encoder,
            T_max=config.NUM_EPOCHS,
            eta_min=1e-6
        )
        
        print("优化器创建完成")
    
    def _initialize_ema(self):
        """初始化指数移动平均参数"""
        print("初始化EMA参数...")
        
        # 初始化均值向量
        self.mu_A = torch.zeros(128).to(self.device)  # A类均值
        self.mu_B = torch.zeros(128).to(self.device)  # B类均值
        
        print("EMA参数初始化完成")
    
    def _load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        if os.path.exists(checkpoint_path):
            print(f"加载检查点: {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.opt_encoder.load_state_dict(checkpoint['opt_encoder_state_dict'])
            self.opt_discriminator.load_state_dict(checkpoint['opt_discriminator_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.start_epoch = checkpoint['epoch']
            self.best_loss = checkpoint.get('best_loss', float('inf'))
            self.mu_A = checkpoint.get('mu_A', self.mu_A)
            self.mu_B = checkpoint.get('mu_B', self.mu_B)
            
            print(f"检查点加载成功，从第 {self.start_epoch} 轮开始")
        else:
            print(f"检查点文件不存在: {checkpoint_path}")
    
    def _save_checkpoint(self, epoch, loss, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'opt_encoder_state_dict': self.opt_encoder.state_dict(),
            'opt_discriminator_state_dict': self.opt_discriminator.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'mu_A': self.mu_A,
            'mu_B': self.mu_B,
            'loss': loss
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(self.logger.get_exp_dir(), 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.logger.get_exp_dir(), 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"保存最佳模型: {best_path}")
    
    def _update_discriminator(self, zA, zB):
        """更新判别器"""
        total_d_loss = 0.0
        
        for _ in range(config.D_STEPS):
            self.opt_discriminator.zero_grad()
            
            # 计算对抗损失
            d_loss = losses.discriminator_loss(self.discriminator, zA, zB)
            
            d_loss.backward()
            self.opt_discriminator.step()
            
            total_d_loss += d_loss.item()
        
        return total_d_loss / config.D_STEPS
    
    def _update_encoder(self, zA, zB, zO):
        """更新编码器"""
        self.opt_encoder.zero_grad()
        
        # 构造负样本
        z_neg = torch.cat([zB, zO], dim=0) if zB.numel() > 0 and zO.numel() > 0 else zB if zB.numel() > 0 else zO
        
        # 计算总损失
        total_loss, loss_dict = losses.total_loss(
            zA, zB, z_neg, self.discriminator, 
            self.mu_A, self.mu_B, config
        )
        
        total_loss.backward()
        self.opt_encoder.step()
        
        return total_loss.item(), loss_dict
    
    def _update_ema_parameters(self, zA, zB):
        """更新EMA参数"""
        if zA.numel() > 0:
            current_mu_A = zA.mean(dim=0)
            self.mu_A = update_ema(self.mu_A, current_mu_A, config.EMA_DECAY)
        
        if zB.numel() > 0:
            current_mu_B = zB.mean(dim=0)
            self.mu_B = update_ema(self.mu_B, current_mu_B, config.EMA_DECAY)
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.encoder.train()
        self.discriminator.train()
        
        epoch_metrics = {
            'loss_total': 0.0,
            'loss_info': 0.0,
            'loss_ncl': 0.0,
            'loss_adv': 0.0,
            'loss_diff': 0.0,
            'd_loss': 0.0
        }
        
        num_batches = 0
        
        # 训练循环
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{config.NUM_EPOCHS}")
        
        for batch_idx, (images, tags) in enumerate(pbar):
            images = images.to(self.device)
            
            # 编码特征
            with torch.no_grad():
                features = self.encoder(images)
            
            # 拆分标签
            zA, zB, zO = split_by_tag(features, tags)
            
            # Step 1: 更新判别器
            if zA.numel() > 0 and zB.numel() > 0:
                zA_detach = zA.detach().requires_grad_(True)
                zB_detach = zB.detach().requires_grad_(True)
                d_loss = self._update_discriminator(zA_detach, zB_detach)
                epoch_metrics['d_loss'] += d_loss
            
            # Step 2: 更新编码器
            if zA.numel() > 0:
                # 重新计算特征（需要梯度）
                features = self.encoder(images)
                zA, zB, zO = split_by_tag(features, tags)
                
                total_loss, loss_dict = self._update_encoder(zA, zB, zO)
                
                # 累计指标
                epoch_metrics['loss_total'] += total_loss
                for key, value in loss_dict.items():
                    if key in epoch_metrics:
                        epoch_metrics[key] += value
            
            # Step 3: 更新EMA参数
            with torch.no_grad():
                features = self.encoder(images)
                zA, zB, zO = split_by_tag(features, tags)
                self._update_ema_parameters(zA, zB)
            
            num_batches += 1
            
            # 更新进度条
            if num_batches % 10 == 0:
                pbar.set_postfix({
                    'Total': f"{epoch_metrics['loss_total']/num_batches:.4f}",
                    'D_loss': f"{epoch_metrics['d_loss']/num_batches:.4f}",
                    '||μA-μB||': f"{torch.norm(self.mu_A - self.mu_B).item():.4f}"
                })
        
        # 计算平均指标
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def train(self):
        """主训练函数"""
        print("=" * 60)
        print("开始训练...")
        print("=" * 60)
        
        for epoch in range(self.start_epoch + 1, config.NUM_EPOCHS + 1):
            # 训练一个epoch
            epoch_metrics = self.train_epoch(epoch)
            
            # 学习率调度
            self.scheduler.step()
            
            # 计算μA和μB之间的距离
            mu_distance = torch.norm(self.mu_A - self.mu_B).item()
            epoch_metrics['mu_distance'] = mu_distance
            epoch_metrics['lr'] = self.scheduler.get_last_lr()[0]
            
            # 记录指标
            self.logger.log_epoch(epoch, epoch_metrics)
            
            # 保存检查点
            is_best = epoch_metrics['loss_total'] < self.best_loss
            if is_best:
                self.best_loss = epoch_metrics['loss_total']
            
            self._save_checkpoint(epoch, epoch_metrics['loss_total'], is_best)
            
            # 可视化
            if epoch % config.VIS_INTERVAL == 0:
                print(f"\n生成第 {epoch} 轮可视化...")
                vis_path = os.path.join(
                    self.logger.get_exp_dir(), 
                    f'tsne_epoch_{epoch:03d}.png'
                )
                
                try:
                    visualize_tsne(
                        self.encoder, 
                        self.eval_loader, 
                        vis_path, 
                        device=self.device,
                        max_samples=1000
                    )
                except Exception as e:
                    print(f"可视化失败: {e}")
            
            # 打印训练信息
            print(f"\nEpoch {epoch}/{config.NUM_EPOCHS} 完成:")
            print(f"  总损失: {epoch_metrics['loss_total']:.4f}")
            print(f"  信息损失: {epoch_metrics['loss_info']:.4f}")
            print(f"  NCL损失: {epoch_metrics['loss_ncl']:.4f}")
            print(f"  对抗损失: {epoch_metrics['loss_adv']:.4f}")
            print(f"  差异损失: {epoch_metrics['loss_diff']:.4f}")
            print(f"  判别器损失: {epoch_metrics['d_loss']:.4f}")
            print(f"  ||μA-μB||₂: {mu_distance:.4f}")
            print(f"  学习率: {epoch_metrics['lr']:.6f}")
            print("-" * 50)
        
        # 训练完成
        print("=" * 60)
        print("训练完成！")
        print("=" * 60)
        
        # 保存最终结果
        self.logger.save_metrics()
        self.logger.plot_losses()
        
        # 生成最终可视化
        final_vis_path = os.path.join(self.logger.get_exp_dir(), 'final_tsne.png')
        try:
            visualize_tsne(
                self.encoder, 
                self.eval_loader, 
                final_vis_path, 
                device=self.device,
                max_samples=2000
            )
            print(f"最终可视化已保存: {final_vis_path}")
        except Exception as e:
            print(f"最终可视化失败: {e}")
        
        print(f"实验结果保存在: {self.logger.get_exp_dir()}")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='A类特征提取训练程序')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS,
                       help=f'训练轮数 (默认: {config.NUM_EPOCHS})')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help=f'设备 (默认: cuda:0)')
    parser.add_argument('--lr-encoder', type=float, default=config.LR_ENCODER,
                       help=f'编码器学习率 (默认: {config.LR_ENCODER})')
    parser.add_argument('--lr-discriminator', type=float, default=config.LR_DISCRIM,
                       help=f'判别器学习率 (默认: {config.LR_DISCRIM})')
    parser.add_argument('--batch-a', type=int, default=config.BATCH_A,
                       help=f'A类批处理大小 (默认: {config.BATCH_A})')
    parser.add_argument('--batch-b', type=int, default=config.BATCH_B,
                       help=f'B类批处理大小 (默认: {config.BATCH_B})')
    parser.add_argument('--batch-o', type=int, default=config.BATCH_O,
                       help=f'O类批处理大小 (默认: {config.BATCH_O})')
    
    # 损失权重
    parser.add_argument('--lambda-info', type=float, default=config.LAMBDA_INFO,
                       help=f'信息损失权重 (默认: {config.LAMBDA_INFO})')
    parser.add_argument('--lambda-ncl', type=float, default=config.LAMBDA_NCL,
                       help=f'NCL损失权重 (默认: {config.LAMBDA_NCL})')
    parser.add_argument('--lambda-adv', type=float, default=config.LAMBDA_ADV,
                       help=f'对抗损失权重 (默认: {config.LAMBDA_ADV})')
    parser.add_argument('--lambda-diff', type=float, default=config.LAMBDA_DIFF,
                       help=f'差异损失权重 (默认: {config.LAMBDA_DIFF})')
    
    # 其他训练参数
    parser.add_argument('--d-steps', type=int, default=config.D_STEPS,
                       help=f'判别器更新步数 (默认: {config.D_STEPS})')
    parser.add_argument('--tau', type=float, default=config.TAU,
                       help=f'对比损失温度参数 (默认: {config.TAU})')
    parser.add_argument('--ema-decay', type=float, default=config.EMA_DECAY,
                       help=f'EMA衰减率 (默认: {config.EMA_DECAY})')
    
    # 其他参数
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--vis-interval', type=int, default=config.VIS_INTERVAL,
                       help=f'可视化间隔 (默认: {config.VIS_INTERVAL})')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (默认: 42)')
    
    return parser.parse_args()


def update_config_from_args(args):
    """根据命令行参数更新配置"""
    if hasattr(config, 'NUM_EPOCHS'):
        config.NUM_EPOCHS = args.epochs
    if hasattr(config, 'LR_ENCODER'):
        config.LR_ENCODER = args.lr_encoder
    if hasattr(config, 'LR_DISCRIM'):
        config.LR_DISCRIM = args.lr_discriminator
    if hasattr(config, 'BATCH_A'):
        config.BATCH_A = args.batch_a
    if hasattr(config, 'BATCH_B'):
        config.BATCH_B = args.batch_b
    if hasattr(config, 'BATCH_O'):
        config.BATCH_O = args.batch_o
    if hasattr(config, 'LAMBDA_INFO'):
        config.LAMBDA_INFO = args.lambda_info
    if hasattr(config, 'LAMBDA_NCL'):
        config.LAMBDA_NCL = args.lambda_ncl
    if hasattr(config, 'LAMBDA_ADV'):
        config.LAMBDA_ADV = args.lambda_adv
    if hasattr(config, 'LAMBDA_DIFF'):
        config.LAMBDA_DIFF = args.lambda_diff
    if hasattr(config, 'D_STEPS'):
        config.D_STEPS = args.d_steps
    if hasattr(config, 'TAU'):
        config.TAU = args.tau
    if hasattr(config, 'EMA_DECAY'):
        config.EMA_DECAY = args.ema_decay
    if hasattr(config, 'VIS_INTERVAL'):
        config.VIS_INTERVAL = args.vis_interval


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 更新配置
    update_config_from_args(args)
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 打印配置信息
    print("=" * 60)
    print("A类特征提取训练程序")
    print("=" * 60)
    print("配置参数:")
    print(f"  训练轮数: {config.NUM_EPOCHS}")
    print(f"  编码器学习率: {config.LR_ENCODER}")
    print(f"  判别器学习率: {config.LR_DISCRIM}")
    print(f"  批处理配置: A={config.BATCH_A}, B={config.BATCH_B}, O={config.BATCH_O}")
    print(f"  损失权重: Info={config.LAMBDA_INFO}, NCL={config.LAMBDA_NCL}, Adv={config.LAMBDA_ADV}, Diff={config.LAMBDA_DIFF}")
    print(f"  判别器更新步数: {config.D_STEPS}")
    print(f"  对比损失温度: {config.TAU}")
    print(f"  EMA衰减率: {config.EMA_DECAY}")
    print(f"  可视化间隔: {config.VIS_INTERVAL}")
    print(f"  随机种子: {args.seed}")
    print("=" * 60)
    
    try:
        # 创建训练器并开始训练
        trainer = FeatureExtractionTrainer(args)
        trainer.train()
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("程序结束")


if __name__ == '__main__':
    main()
