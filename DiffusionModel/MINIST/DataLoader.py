import sys
sys.path.append('/Users/dahudieyu/Documents/Education/MyPython/DigitalImagePeocessing/DiffusionModel/MINIST')

from Config import *
# MNIST 数据
transform = transforms.Compose([
    transforms.ToTensor(),  # (0, 1)
    transforms.Lambda(lambda x: x * 2 - 1)  # 缩放到 (-1, 1)
])
dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

