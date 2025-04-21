import sys
sys

# 假设你有一个 batch_size 为 8 的图像噪声（例如随机噪声），形状为 [8, 3, 64, 64]
# 还有来自 CLIP 文本编码器的文本嵌入，形状为 [8, 768]
from unet_pytorch import UNet
from text_encoder import TextEncoder
import torch
unet = UNet()

# 随机生成输入数据
texts = ["a cat on a bed"]
noise = torch.randn(1, 3, 64, 64)  # 随机噪声图像
text_encoder = TextEncoder()
text_embeddings = text_encoder(texts)
print(text_embeddings.shape)


# 生成结果
generated_image = unet(noise, text_embeddings)
print(generated_image.shape)  # 应该是 [1, 3, 64, 64]
