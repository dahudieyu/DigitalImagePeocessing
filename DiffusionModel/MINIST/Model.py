import sys
sys.path.append('/Users/dahudieyu/Documents/Education/MyPython/DigitalImagePeocessing/DiffusionModel/MINIST')

from Config import *


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
        t_embed = t_embed.expand(-1, 1, x.shape[2], x.shape[3])
        x_input = torch.cat([x, t_embed], dim=1)
        return self.net(x_input)
    
model = SimpleUnet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)