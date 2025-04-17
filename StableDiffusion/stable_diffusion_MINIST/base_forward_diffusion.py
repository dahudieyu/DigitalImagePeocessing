from config import *


# 一维情况下进行N步前向扩散
def forward_diffusion_1D(x0, noise_strength_fn, t0, nsteps, dt):
    """
    参数:
    - x0: 初始样本值（标量）
    - noise_strength_fn: 噪声强度函数，随时间变化，输出标量噪声强度
    - t0: 初始时间
    - nsteps: 扩散步数
    - dt: 时间步长

    返回:
    - x: 样本值随时间的轨迹
    - t: 轨迹对应的时间点
    """

    # 初始化轨迹数组
    x = np.zeros(nsteps + 1)
    print("x : ", x)
    
    # 设置初始样本值
    x[0] = x0
    print("x[0] : ", x[0])

    # 生成轨迹的时间点
    t = t0 + np.arange(nsteps + 1) * dt
    print("np.arange(nsteps + 1) :" , np.arange(nsteps + 1))
    print("t : ", t)

    # 执行Euler-Maruyama时间步进行扩散模拟
    for i in range(nsteps):

        # 获取当前时间的噪声强度
        noise_strength = noise_strength_fn(t[i])

        # 生成一个随机正态变量
        random_normal = np.random.randn()

        # 使用Euler-Maruyama方法更新轨迹
        x[i + 1] = x[i] + random_normal * noise_strength

    # 返回轨迹和对应的时间点
    return x, t

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

# 我们已经定义了前向扩散组件，现在让我们检查它在不同试验中的工作情况。

# 扩散步数
nsteps = 100

# 初始时间
t0 = 0

# 时间步长
dt = 0.1

# 噪声强度函数
noise_strength_fn = noise_strength_constant

# 初始样本值
x0 = 0

# 可视化的试验次数
num_tries = 5

# 设置图的宽度较大，高度较小
plt.figure(figsize=(15, 5))

# 设置字体 windows 下需要设置
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# mac
plt.rcParams['font.family'] = 'Arial Unicode MS'  # macOS 自带字体
# 或者使用苹果方黑
plt.rcParams['font.family'] = 'PingFang HK'  
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 多次试验循环
for i in range(num_tries):

    # 模拟前向扩散
    x, t = forward_diffusion_1D(x0, noise_strength_fn, t0, nsteps, dt)

    # 绘制轨迹
    plt.plot(t, x, label=f'试验 {i+1}')  # 为每次试验添加标签

# 给图形添加标签
plt.xlabel('时间', fontsize=20)
plt.ylabel('样本值 ($x$)', fontsize=20)

# 图的标题
plt.title('前向扩散可视化', fontsize=20)

# 添加图例以区分每次试验
plt.legend()

# 显示图形
plt.show()
