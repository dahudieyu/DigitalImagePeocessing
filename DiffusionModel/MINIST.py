import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os
import numpy as np
from tqdm import tqdm


# 超参数
image_size = 28 # 这里的MINIST图片大小为28*28
batch_size = 128 # 训练批次  每次训练 128 张图片
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
T = 300  # 扩散步数

# MNIST 数据
transform = transforms.Compose([
    transforms.ToTensor(),  # (0, 1)
    transforms.Lambda(lambda x: x * 2 - 1)  # 缩放到 (-1, 1)
])
dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True) # 下载MNIST数据集 并且使用transform规范化图片
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True) # 加载数据 批次大小为 128 并且打乱顺序



class SimpleUnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1 + 1, 64, 3, padding=1),  # 图像 + 时间
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1),
        )

    def forward(self, x, t):
        # 将 t 扩展成和图像一样的大小
        t_embed = t[:, None, None, None].float() / T
        t_embed = t_embed.expand(-1, 1, x.shape[2], x.shape[3]) # 扩展成和图像一样的大小 128 1 28 28
        x_input = torch.cat([x, t_embed], dim=1)
        return self.net(x_input)

def q_sample(x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    alpha_bar = torch.exp(-0.02 * t.float().view(-1, 1, 1, 1))  # .view(-1, 1, 1, 1) 是为了让每张图的时间步 t 转换成能和图像 [128, 1, 28, 28] 对应的形状，方便进行逐元素的广播运算。  exp 是做以e为底的指数运算
    return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise

model = SimpleUnet().to(device) # 定义模型 
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 10

for epoch in range(epochs):
    for x0, _ in tqdm(dataloader): # tdqm 是一个进度条 显示训练进度  x0 是 128 张图片  _ 是标签 比如：5 
        x0 = x0.to(device) # 将训练的图片放到mps（因为我是mac）
        t = torch.randint(1, T, (x0.shape[0],), device=device) # 随机生成 1 到 T 之间的整数 作为时间步 128 张图片 对应 128 个时间步
        noise = torch.randn_like(x0) # torch.randn_like：返回一个张量，其形状和输入张量相同，且元素服从标准正态分布。 
        xt = q_sample(x0, t, noise) # 

        pred_noise = model(xt, t)
        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1} - Loss: {loss.item():.4f}")

@torch.no_grad() # 
def sample(model, steps=T):
    model.eval()
    x = torch.randn(1, 1, 28, 28).to(device)

    for t_ in reversed(range(1, steps)):
        t = torch.full((x.size(0),), t_, device=device)
        predicted_noise = model(x, t)
        alpha_bar = torch.exp(-0.02 * t.float().view(-1, 1, 1, 1))
        x = (x - torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha_bar)
        if t_ > 1:
            x += torch.randn_like(x) * 0.1  # 少量随机扰动

    return x

samples = sample(model)
samples = (samples + 1) / 2  # 还原到 (0, 1)
save_image(samples, 'samples.png', nrow=8)
# 运行完这个代码后，会在当前目录下生成一张 samples.png，你可以看到生成出来的“手写数字图像”。


# torch.randint 的用法：
# torch.randint(low, high, size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
# 功能：返回一个张量，包含在 low 和 high 之间均匀分布的 size 个随机整数。

#torch.randn_like 的用法：
#torch.randn_like(input, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
#功能：返回一个张量，其形状和 input 相同，且元素服从标准正态分布。

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision
# import torchvision.transforms as transforms
# from torchvision.utils import save_image
# from tqdm import tqdm
# import numpy as np
# import os

# # 训练设备
# device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# # 超参数
# image_size = 28
# batch_size = 128
# T = 300
# epochs = 10
# lr = 1e-4

# # 数据处理
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Lambda(lambda x: x * 2 - 1)
# ])
# dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# # 时间位置嵌入模块
# class SinusoidalPositionEmbeddings(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim
        

#     def forward(self, time):
#         device = time.device
#         half_dim = self.dim // 2
#         emb = torch.exp(torch.arange(half_dim, device=device) * -(np.log(10000) / (half_dim - 1)))
#         emb = time[:, None] * emb[None, :]
#         emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
#         return emb

# # 更完整的 UNet 网络
# class UNet(nn.Module):
#     def __init__(self):
#         super(UNet, self).__init__()
#         self.time_mlp = nn.Sequential(
#             SinusoidalPositionEmbeddings(32),
#             nn.Linear(32, 64),
#             nn.ReLU()
#         )
#         self.time_emb_proj = nn.Conv2d(64, 128, 1)


#         # 编码器
#         self.down1 = nn.Sequential(nn.Conv2d(1, 64, 3, padding=1), nn.ReLU())
#         self.down2 = nn.Sequential(nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU())
#         self.down3 = nn.Sequential(nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU())

#         # 中间层
#         self.middle = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())

#         # 解码器
#         self.up1 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU())
#         self.up2 = nn.Sequential(nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1), nn.ReLU())
#         self.out = nn.Conv2d(128, 1, 1)
        

#     def forward(self, x, t):
#         t_emb = self.time_mlp(t)[:, :, None, None]  # shape: [B, C, 1, 1]  【128，64，1，1】
#          # 让 t_emb 尺寸和 x2 匹配
#         t_emb2 = F.interpolate(t_emb, size=(14, 14), mode='nearest')
#         t_emb2 = self.time_emb_proj(t_emb2)  # [B, 128, 14, 14]

#         x1 = self.down1(x)                          # [B, 64, 28, 28]  【128，64，28，28】
#         x2 = self.down2(x1)                         # [B, 128, 14, 14]
#         x3 = self.down3(x2 + t_emb2)                # [B, 256, 7, 7]
        
#         mid = self.middle(x3)

#         u1 = self.up1(mid)                          # [B, 128, 14, 14]
#         u1 = torch.cat([u1, x2], dim=1)             # 拼接
#         u2 = self.up2(u1)                           # [B, 64, 28, 28]
#         u2 = torch.cat([u2, x1], dim=1)

#         return self.out(u2)                         # [B, 1, 28, 28]


# # 正向扩散过程
# def q_sample(x0, t, noise=None):
#     if noise is None:
#         noise = torch.randn_like(x0)
#     alpha_bar = torch.exp(-0.02 * t.float().view(-1, 1, 1, 1)) # t[128，1，1，1]
#     return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise

# # 模型训练
# model = UNet().to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# for epoch in range(epochs):
#     for x0, _ in tqdm(dataloader):
#         x0 = x0.to(device) # x0[128, 1, 28, 28]
#         t = torch.randint(1, T, (x0.shape[0],), device=device) # t[128]
#         noise = torch.randn_like(x0)
#         xt = q_sample(x0, t, noise) # xt[128, 1, 28, 28]

#         pred_noise = model(xt, t)
#         loss = F.mse_loss(pred_noise, noise)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

# # 采样函数
# @torch.no_grad()
# def sample(model, steps=T):
#     model.eval()
#     x = torch.randn(1, 1, 28, 28).to(device)

#     for t_ in reversed(range(1, steps)):
#         t = torch.full((x.size(0),), t_, device=device)
#         predicted_noise = model(x, t)
#         alpha_bar = torch.exp(-0.02 * t.float().view(-1, 1, 1, 1))
#         x = (x - torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha_bar)
#         if t_ > 1:
#             x += torch.randn_like(x) * 0.1
#     return x

# # 生成图像
# samples = sample(model)
# samples = (samples + 1) / 2  # 还原到 (0,1)
# save_image(samples, 'samples.png', nrow=8)
# print("✅ 已保存生成图像 samples.png")
