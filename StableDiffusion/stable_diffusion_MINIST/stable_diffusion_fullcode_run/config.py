device = "cuda"

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 必须放在所有import之前

# 导入用于张量操作的PyTorch库。
import torch

# 从PyTorch导入神经网络模块。
import torch.nn as nn

# 从PyTorch导入功能操作。
import torch.nn.functional as F

# 导入用于数值运算的'numpy'库。
import numpy as np

# 导入用于高阶函数的'functools'模块。
import functools

# 从PyTorch导入Adam优化器。
from torch.optim import Adam

# 从PyTorch导入DataLoader类以处理数据集。
from torch.utils.data import DataLoader

# 从torchvision导入数据变换函数。
import torchvision.transforms as transforms

# 从torchvision导入MNIST数据集。
from torchvision.datasets import MNIST

# 导入用于在训练过程中创建进度条的'tqdm'库。
import tqdm

# 特别为笔记本兼容性导入'trange'和'tqdm'。
from tqdm.notebook import trange, tqdm

# 从PyTorch导入学习率调度器。
from torch.optim.lr_scheduler import MultiplicativeLR, LambdaLR

# 导入用于绘制图形的'matplotlib.pyplot'库。
import matplotlib.pyplot as plt

# 从torchvision.utils导入'make_grid'函数以可视化图像网格。
from torchvision.utils import make_grid

# 从'einops'库导入'rearrange'函数。
from einops import rearrange

# 导入'math'模块以进行数学运算。
import math


# 在代码开头添加
torch.backends.cudnn.enabled = False  # 禁用 cuDNN 优化
