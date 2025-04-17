# 稳定扩散通过从完全随机的图像开始创建图像。然后，噪声预测器猜测图像的噪声程度，并从图像中移除该猜测的噪声。这个循环重复多次，最终产生一个干净的图像。

# 这种清理过程被称为“采样”，因为稳定扩散在每个步骤中都会生成一个新的图像样本。创建这些样本的方法称为“采样器”或“采样方法”。

from config import *
# 采样步骤数
num_steps = 500

def Euler_Maruyama_sampler(score_model,
                           marginal_prob_std,
                           diffusion_coeff,
                           batch_size=64,
                           x_shape=(1, 28, 28),
                           num_steps=num_steps,
                           device='cuda',
                           eps=1e-3, y=None):
    """
    使用Euler-Maruyama求解器从基于得分的模型生成样本。

    参数：
    - score_model: 表示时间相关的基于得分的模型的PyTorch模型。
    - marginal_prob_std: 提供扰动核的标准差的函数。
    - diffusion_coeff: 提供SDE的扩散系数的函数。
    - batch_size: 每次调用该函数生成的采样数。
    - x_shape: 样本的形状。
    - num_steps: 采样步骤数，相当于离散化的时间步数。
    - device: 'cuda'表示在GPU上运行，'cpu'表示在CPU上运行。
    - eps: 数值稳定性的最小时间步。
    - y: 目标张量（在此函数中未使用）。

    返回：
    - 样本。
    """

    # 初始化时间和初始样本
    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, *x_shape, device=device) * marginal_prob_std(t)[:, None, None, None]
    
    # 生成时间步
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    
    # 使用Euler-Maruyama方法采样
    with torch.no_grad():
        for time_step in tqdm(time_steps):
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            g = diffusion_coeff(batch_time_step)
            mean_x = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step, y=y) * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)
    
    # 最后的采样步骤中不包含任何噪声。
    return mean_x
