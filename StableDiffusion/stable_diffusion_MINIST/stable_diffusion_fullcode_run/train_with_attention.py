from config import *
from Unet_Tranformer import *
from e_noise_forward_diffusion import *
from loss_function_condition import *




# 指定是否继续训练或初始化新模型
continue_training = False # 设置为 True 或 False

if not continue_training:
    # 初始化一个新的带 Transformer 的 UNet 模型
    score_model = torch.nn.DataParallel(UNet_Tranformer(marginal_prob_std=marginal_prob_std_fn))
    score_model = score_model.to(device)

# 设置训练超参数
n_epochs =   100   # {'type':'integer'}
batch_size =  1024 # {'type':'integer'}
lr = 10e-4         # {'type':'number'}

# 加载 MNIST 数据集并创建数据加载器
dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# 定义优化器和学习率调度器
optimizer = Adam(score_model.parameters(), lr=lr)
scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: max(0.2, 0.98 ** epoch))

# 使用 tqdm 显示 epoch 的进度条
tqdm_epoch = trange(n_epochs)
for epoch in tqdm_epoch:
    avg_loss = 0.
    num_items = 0

    # 遍历数据加载器中的批次
    for x, y in tqdm(data_loader):
        x = x.to(device)

        # 使用条件得分模型计算损失
        loss = loss_fn_cond(score_model, x, y, marginal_prob_std_fn)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]

    # 使用调度器调整学习率
    scheduler.step()
    lr_current = scheduler.get_last_lr()[0]

    # 打印 epoch 信息，包括平均损失和当前学习率
    print('{} Average Loss: {:5f} lr {:.1e}'.format(epoch, avg_loss / num_items, lr_current))
    tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))

    # 在每个 epoch 结束后保存模型检查点
    torch.save(score_model.state_dict(), 'ckpt_transformer.pth')
