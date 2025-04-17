

from config import *
from  Unet_with_skip_connection import UNet
from e_noise_forward_diffusion import *
from loss_function import *

# 定义基于得分的模型并将其移动到指定设备
score_model = torch.nn.DataParallel(UNet(marginal_prob_std=marginal_prob_std_fn))
score_model = score_model.to(device)

# 训练epoch数
n_epochs = 50
# 小批量大小
batch_size = 2048
# 学习率
lr = 5e-4

# 加载MNIST数据集并创建数据加载器
dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# 定义Adam优化器来训练模型
optimizer = Adam(score_model.parameters(), lr=lr)

# epoch的进度条
tqdm_epoch = trange(n_epochs)
for epoch in tqdm_epoch:
    avg_loss = 0.
    num_items = 0
    # 迭代数据加载器中的小批量数据
    for x, y in tqdm(data_loader):
        x = x.to(device)
        # 计算损失并执行反向传播
        loss = loss_fn(score_model, x, marginal_prob_std_fn)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]
    # 打印当前epoch的平均训练损失
    tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
    # 在每个训练epoch后保存模型检查点
    torch.save(score_model.state_dict(), 'ckpt.pth')
