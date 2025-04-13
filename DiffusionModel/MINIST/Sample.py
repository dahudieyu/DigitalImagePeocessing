import sys
sys.path.append('/Users/dahudieyu/Documents/Education/MyPython/DigitalImagePeocessing/DiffusionModel/MINIST')

from Config import *
from Model import *



@torch.no_grad()
def sample(model, steps=T):
    model.eval() # 切换到测试模式
    x = torch.randn(64, 1, 28, 28).to(device)

    for t_ in reversed(range(1, steps)):
        t = torch.full((x.size(0),), t_, device=device)
        predicted_noise = model(x, t)
        alpha_bar = torch.exp(-0.02 * t.float().view(-1, 1, 1, 1))
        x = (x - torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha_bar)
        if t_ > 1:
            x += torch.randn_like(x) * 0.1  # 少量随机扰动

    return x

# 创建一个新的模型实例
model = SimpleUnet().to(device)

# 加载保存的模型参数
model.load_state_dict(torch.load('simple_unet_model.pth'))
samples = sample(model)
samples = (samples + 1) / 2  # 还原到 (0, 1)
save_image(samples, 'samples.png', nrow=8)
# 运行完这个代码后，会在当前目录下生成一张 samples.png，你可以看到生成出来的“手写数字图像”。