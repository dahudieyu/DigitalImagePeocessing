from config import *
from time_embedding_net import *

# 定义一个基于U-Net架构的时间依赖评分模型
class UNet(nn.Module):
    def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256):
        """
        初始化一个时间依赖的评分网络。

        参数：
        - marginal_prob_std：一个函数，接受时间t并给出扰动核p_{0t}(x(t) | x(0))的标准差。
        - channels：每个分辨率的特征图通道数。
        - embed_dim：高斯随机特征嵌入的维度。
        """

        super().__init__()

        # 时间的高斯随机特征嵌入层
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )

        # 分辨率降低的编码层
        self.conv1 = nn.Conv2d(1, channels[0], kernel_size=3, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])

        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])

        # 额外的编码层
        self.conv3 = nn.Conv2d(channels[1], channels[2],kernel_size= 3, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])

        self.conv4 = nn.Conv2d(channels[2], channels[3],kernel_size= 3, stride=2, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        # 分辨率增加的解码层
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=3, stride=2, bias=False)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])

        self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], kernel_size=3, stride=2, bias=False, output_padding=1)
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
 
        self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], kernel_size=3, stride=2, bias=False, output_padding=1)
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])

        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, kernel_size=3, stride=1)

        # Swish激活函数
        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t, y=None):
        """
        参数：
        - x：输入张量
        - t：时间张量
        - y：目标张量（在此前向传递中未使用）

        返回：
        - h：经过U-Net架构处理后的输出张量
        """



        # 获取时间t的高斯随机特征嵌入
        embed = self.act(self.time_embed(t))

        # 编码路径
        h1 = self.conv1(x) + self.dense1(embed)
        print("h1: ", h1)
        print("h1 shape:", h1.shape)
        print("self.conv1:", self.conv1(x).shape)
        print("self.dense1(embed):", self.dense1(embed).shape)
        print("self.gnorm1 shape:", self.gnorm1(h1).shape)
        print("self.gnorm1 :", self.gnorm1(h1))
        print("self.act(self.gnorm1(h1)) shape:", self.act(self.gnorm1(h1)).shape)
        print("self.act(self.gnorm1(h1)):", self.act(self.gnorm1(h1)))
        print("---------------------")
        h1 = self.act(self.gnorm1(h1))
        h2 = self.conv2(h1) + self.dense2(embed)
        h2 = self.act(self.gnorm2(h2))

        # 额外的编码路径层
        h3 = self.conv3(h2) + self.dense3(embed)
        h3 = self.act(self.gnorm3(h3))
        h4 = self.conv4(h3) + self.dense4(embed)
        h4 = self.act(self.gnorm4(h4))

        # 解码路径
        h = self.tconv4(h4)
        h += self.dense5(embed)
        h = self.act(self.tgnorm4(h))
        print("h4:", h4.shape) #h4: torch.Size([1, 256, 2, 2])
        print("h:", h.shape) # h: torch.Size([1, 128, 5, 5])

        h = self.tconv3(torch.cat([h, h3], dim=1))
        print("h3:", h3.shape) #h3: torch.Size([1, 256, 2, 2])
        h += self.dense6(embed)
        h = self.act(self.tgnorm3(h))
        print("h3:", h3.shape) #h3: torch.Size([1, 128, 5, 5])
        print("h:", h.shape) # h: torch.Size([1, 64, 10, 10])

        h = self.tconv2(torch.cat([h, h2], dim=1))
        h += self.dense7(embed)
        h = self.act(self.tgnorm2(h))

        h = self.tconv1(torch.cat([h, h1], dim=1))
        print(" before norm h:", h.shape)

        # 归一化输出
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        print("after norm h:", h.shape)
        return h

if __name__ == '__main__':
    # 定义一个测试用的时间依赖评分网络
    model = UNet(marginal_prob_std=lambda t: torch.ones(t.shape[0]), embed_dim=128)
    x = torch.randn(1, 1, 28, 28)
    t = torch.randn(1)
    y = torch.randint(0, 10, (1,))
    h = model(x, t, y)
    print(h.shape)
    print("h:", h)