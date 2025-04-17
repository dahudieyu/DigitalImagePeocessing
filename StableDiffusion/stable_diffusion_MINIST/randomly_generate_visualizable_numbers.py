from config import *
from StableDiffusion.stable_diffusion_MINIST.train_with_Unet import *
from sampler import *

#从磁盘加载预训练的检查点。
device = 'cuda'

# 加载预训练的模型检查点
ckpt = torch.load('ckpt.pth', map_location=device)
score_model.load_state_dict(ckpt)

# 设置采样批量大小和步骤数
sample_batch_size = 64
num_steps = 500

# 选择Euler-Maruyama采样器
sampler = Euler_Maruyama_sampler

# 使用指定的采样器生成样本
samples = sampler(score_model,
                  marginal_prob_std_fn,
                  diffusion_coeff_fn,
                  sample_batch_size,
                  num_steps=num_steps,
                  device=device,
                  y=None)

# 将样本裁剪到范围[0, 1]
samples = samples.clamp(0.0, 1.0)

# 可视化生成的样本
# matplotlib inline
import matplotlib.pyplot as plt
sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))

# 绘制样本网格
plt.figure(figsize=(6, 6))
plt.axis('off')
plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
plt.show()
