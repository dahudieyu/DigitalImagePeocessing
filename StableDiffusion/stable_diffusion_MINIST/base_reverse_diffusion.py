from config import *

# 该反向扩散实现的数学公式：
# score-based diffusion models，它建立在随机微分方程（SDE）基础上，通过数值解反向SDE实现生成，这种方法由杨立昆组的论文《Score-Based Generative Modeling through SDEs》（2021）推广

# 一维反扩散N步。
def reverse_diffusion_1D(x0, noise_strength_fn, score_fn, T, nsteps, dt):
    """
    参数:
    - x0: 初始样本值（标量）
    - noise_strength_fn: 时间的函数，输出标量噪声强度
    - score_fn: 分数函数
    - T: 最终时间
    - nsteps: 扩散步数
    - dt: 时间步长

    返回值:
    - x: 样本值随时间变化的轨迹
    - t: 轨迹对应的时间点
    """

    # 初始化轨迹数组
    x = np.zeros(nsteps + 1)
    
    # 设置初始样本值
    x[0] = x0

    # 生成轨迹的时间点
    t = np.arange(nsteps + 1) * dt

    # 进行反扩散模拟的Euler-Maruyama时间步长
    for i in range(nsteps):

        # 计算当前时间的噪声强度
        noise_strength = noise_strength_fn(T - t[i])

        # 使用分数函数计算分数
        score = score_fn(x[i], 0, noise_strength, T - t[i])

        # 生成一个随机正态变量
        random_normal = np.random.randn()

        # 使用反向Euler-Maruyama方法更新轨迹
        x[i + 1] = x[i] + score * noise_strength**2 * dt + noise_strength * random_normal * np.sqrt(dt)

    # 返回轨迹和对应的时间点
    return x, t

# 示例分数函数: 总是等于1
def score_simple(x, x0, noise_strength, t):
    """
    参数:
    - x: 当前样本值（标量）
    - x0: 初始样本值（标量）
    - noise_strength: 当前时间的标量噪声强度
    - t: 当前时间

    返回值:
    - score: 根据提供的公式计算的分数
    """

    # 使用提供的公式计算分数
    score = - (x - x0) / ((noise_strength**2) * t)

    # 返回计算的分数
    return score

# 噪声强度函数始终等于1的示例
def noise_strength_constant(t):
    """
    示例噪声强度函数，返回一个常数值（1）。

    参数:
    - t: 时间参数（在此示例中未使用）

    返回:
    - 常数噪声强度（1）
    """
    return 1

# 反扩散步数
nsteps = 100

# 反扩散的初始时间
t0 = 0

# 反扩散的时间步长
dt = 0.1

# 定义常数噪声强度的函数用于反扩散
noise_strength_fn = noise_strength_constant

# 反扩散的示例分数函数
score_fn = score_simple

# 反扩散的初始样本值
x0 = 0

# 反扩散的最终时间
T = 11

# 可视化的尝试次数
num_tries = 5

# 设置较宽的图形宽度和较小的高度
plt.figure(figsize=(15, 5))

plt.rcParams['font.family'] = 'Arial Unicode MS'  # macOS 自带字体
# 或者使用苹果方黑
plt.rcParams['font.family'] = 'PingFang HK'  
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 多次尝试的循环
for i in range(num_tries):
    # 从噪声分布中抽取，该分布是噪声强度为1时扩散时间为T的分布
    x0 = np.random.normal(loc=0, scale=T)

    # 模拟反扩散
    x, t = reverse_diffusion_1D(x0, noise_strength_fn, score_fn, T, nsteps, dt)

    # 绘制轨迹
    plt.plot(t, x, label=f'Trial {i+1}')  # 为每次尝试添加标签

# 图表标签
plt.xlabel('时间', fontsize=20)
plt.ylabel('样本值 ($x$)', fontsize=20)

# 图表标题
plt.title('反扩散可视化', fontsize=20)

# 添加图例以标识每次尝试
plt.legend()

# 显示图表
plt.show()
