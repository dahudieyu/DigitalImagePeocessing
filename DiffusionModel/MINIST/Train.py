import sys
sys.path.append('/Users/dahudieyu/Documents/Education/MyPython/DigitalImagePeocessing/DiffusionModel/MINIST')

from Config import *
from DataLoader import *
from Model import *

def q_sample(x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    alpha_bar = torch.exp(-0.02 * t.float().view(-1, 1, 1, 1))  # 模拟 β 序列
    return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise


for epoch in range(epochs):
    for x0, _ in tqdm(dataloader):
        x0 = x0.to(device)
        t = torch.randint(1, T, (x0.shape[0],), device=device)
        noise = torch.randn_like(x0)
        xt = q_sample(x0, t, noise)

        pred_noise = model(xt, t)
        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1} - Loss: {loss.item():.4f}")

# 保存模型
torch.save(model.state_dict(), 'simple_unet_model.pth')