"""
实验全局配置文件
集中管理与"提取A类相对于B类独特特征"实验相关的所有超参数和常量
方便统一修改与复现实验
"""

# ======================== 数据配置 ========================
# CIFAR-10 数据根目录
CIFAR_DATA_PATH = '/home/mjt2024/unlearning/SimCLR_Pytorch/Dataset/cifar-10-python.tar.gz'

# 遗忘集索引文件路径
FORGET_INDICES_FILE = '/home/mjt2024/unlearning/SimCLR_Pytorch/forget_indices.json'

# ======================== 数据集大小配置 ========================
# 遗忘集大小（A类）
A_SIZE = 50

# 保留集大小（B类）
B_SIZE = 4950

# ======================== 批处理配置 ========================
# A类批处理大小
BATCH_A = 24

# B类批处理大小
BATCH_B = 128

# 其他数据批处理大小
BATCH_O = 120

# ======================== 训练超参数 ========================
# 训练轮数
NUM_EPOCHS = 100

# 编码器学习率
LR_ENCODER = 1e-4

# 判别器学习率
LR_DISCRIM = 5e-4

# 权重衰减
WEIGHT_DECAY = 1e-4

# 判别器更新步数
D_STEPS = 4

# EMA衰减率
EMA_DECAY = 0.9

# 可视化间隔
VIS_INTERVAL = 10

# ======================== 对比学习配置 ========================
# 对比损失温度参数
TAU = 0.15

# ======================== 损失权重配置 ========================
# 信息损失权重
LAMBDA_INFO = 1.0

# 负对比学习损失权重
LAMBDA_NCL = 1.0

# 对抗损失权重
LAMBDA_ADV = 0.8

# 差异损失权重
LAMBDA_DIFF = 0.5

# ======================== 模型配置 ========================
# 骨干网络模型名称
BACKBONE = 'vit_tiny_patch16_224'

# 冻结层数
FREEZE_BLOCKS = 6
