from config import *

# 使用GPU
# device = "cuda"
# device = "mps"

# 边际概率标准差函数
def marginal_prob_std(t, sigma):
    """
    计算 $p_{0t}(x(t) | x(0))$ 的均值和标准差。

    参数：
    - t: 时间步向量。
    - sigma: SDE 中的 $\sigma$。

    返回：
    - 标准差。
    """
    # 将时间步转换为PyTorch张量
    t = torch.tensor(t, device=device)
    
    # 根据给定公式计算并返回标准差
    return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))


def diffusion_coeff(t, sigma):
    """
    计算SDE的扩散系数。

    参数：
    - t: 时间步向量。
    - sigma: SDE 中的 $\sigma$。

    返回：
    - 扩散系数向量。
    """
    # 根据给定公式计算并返回扩散系数
    return torch.tensor(sigma**t, device=device)


# Sigma值
sigma = 25.0

# 边际概率标准差
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)

# 扩散系数
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

