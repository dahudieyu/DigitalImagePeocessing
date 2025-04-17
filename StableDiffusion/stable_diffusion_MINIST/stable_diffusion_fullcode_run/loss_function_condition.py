from config import *

def loss_fn_cond(model, x, y, marginal_prob_std, eps=1e-5):
    """使用条件信息训练得分生成模型的损失函数。

    参数:
    - model: 表示时间依赖得分模型的 PyTorch 模型实例。
    - x: 一小批训练数据。
    - y: 条件信息（目标张量）。
    - marginal_prob_std: 一个函数，返回扰动核的标准差。
    - eps: 数值稳定性的容差值。

    返回:
    - loss: 计算出的损失。
    """
    # 在范围 [eps, 1-eps] 内均匀采样时间
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
    # 生成与输入形状相同的随机噪声
    z = torch.randn_like(x)
    # 计算采样时间下扰动核的标准差
    std = marginal_prob_std(random_t)
    # 用生成的噪声和标准差扰动输入数据
    perturbed_x = x + z * std[:, None, None, None]
    # 获取模型对扰动输入的得分，考虑条件信息
    score = model(perturbed_x, random_t, y=y)
    # 使用得分和扰动计算损失
    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1, 2, 3)))
    return loss
