# from config import *
# from time_embedding_net import *

# # nsteps = 100
# # x = np.zeros(nsteps + 1)
# # print(x)

# # print(np.arange(nsteps + 1)*0.1)、
# # class UNet(nn.Module):
# #     def __init__(self, embed_dim = 256) :
# #         super().__init__()
# #         # 时间的高斯随机特征嵌入层
# #         self.time_embed = nn.Sequential(
# #             GaussianFourierProjection(embed_dim=embed_dim),
# #             nn.Linear(embed_dim, embed_dim)
# #         )
# #         self.act = lambda x: x * torch.sigmoid(x)
        
# #     def forward(self, t):
# #         print("time embedding", self.time_embed(t))
# #              # 获取时间t的高斯随机特征嵌入
# #         embed = self.act(self.time_embed(t))
# #         return embed
        
# # if __name__ == '__main__':
# #     model = UNet()
# #     m = model(torch.tensor([1], dtype=torch.float32))
# #     print(m)
# #     print(m.shape) # [1, 256]


# y = torch.randint(0, 10, (1,))
# print(y)


import torch, torchvision
print(torch.__version__)       # 应该输出1.13.1
print(torchvision.__version__) # 应该输出0.14.1


import torch
print("是否支持CUDA:", torch.cuda.is_available())
print("CUDA设备数量:", torch.cuda.device_count())
print("当前CUDA设备名称:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无")
