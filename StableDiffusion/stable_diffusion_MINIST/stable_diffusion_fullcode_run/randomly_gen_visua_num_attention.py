
from config import *
from train_with_attention import *
from sampler import *

if __name__ == '__main__':
        # 从磁盘加载预训练的检查点。
        import os
        print("当前工作目录:", os.getcwd())  # 查看实际保存路径


        # device = 'cuda' #@param ['cuda', 'cpu'] {'type':'string'}
        ckpt = torch.load('ckpt_transformer.pth', map_location=device)
        print("Debug: score_model defined?", 'score_model' in locals())  # 检查变量是否存在

        score_model.load_state_dict(ckpt)
        print("Debug: score_model defined?", 'score_model' in locals())  # 检查变量是否存在


        #指定生成样本的数字
        ###########
        digit = 8 #@param {'type':'integer'}

        # 设置生成样本的批量大小
        sample_batch_size = 64 #@param {'type':'integer'}
        # 设置Euler-Maruyama采样器的步数
        num_steps = 250 #@param {'type':'integer'}
        # 选择采样器类型（Euler-Maruyama, pc_sampler, ode_sampler）
        sampler = Euler_Maruyama_sampler #@param ['Euler_Maruyama_sampler', 'pc_sampler', 'ode_sampler'] {'type': 'raw'}
        # score_model.eval()

        ## 使用指定的采样器生成样本。
        samples = sampler(score_model,
                marginal_prob_std_fn,
                diffusion_coeff_fn,
                sample_batch_size,
                num_steps=num_steps,
                device=device,
                y=digit*torch.ones(sample_batch_size, dtype=torch.long))

        ## 样本可视化。
        samples = samples.clamp(0.0, 1.0)
        # %matplotlib inline
        import matplotlib.pyplot as plt
        # 创建样本网格以便可视化
        sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))

        # 绘制生成的样本
        plt.figure(figsize=(6,6))
        plt.axis('off')
        plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
         # 保存生成的图片
        plt.savefig('./generated_samples.png', bbox_inches='tight', pad_inches=0)
        plt.show()
       
