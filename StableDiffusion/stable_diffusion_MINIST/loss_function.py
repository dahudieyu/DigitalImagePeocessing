from config import *


def loss_fn(model, x, marginal_prob_std, eps=1e-5):
    """
    用于训练基于得分的生成模型的损失函数。

    参数：
    - model: 表示时间相关的基于得分的模型的PyTorch模型实例。
    - x: 训练数据的小批量。
    - marginal_prob_std: 提供扰动核的标准差的函数。
    - eps: 数值稳定性的容差值。
    """
    # 在范围(eps, 1-eps)内均匀采样时间
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - 2 * eps) + eps
    # 在采样时间`t`找到噪声标准差
    std = marginal_prob_std(random_t)
    
    # 生成正态分布的噪声
    z = torch.randn_like(x)
    
    # 使用生成的噪声扰动输入数据
    perturbed_x = x + z * std[:, None, None, None]
    
    # 使用扰动数据和时间从模型获取得分
    score = model(perturbed_x, random_t)
    
    # 基于得分和噪声计算损失
    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1, 2, 3)))
    
    return loss
