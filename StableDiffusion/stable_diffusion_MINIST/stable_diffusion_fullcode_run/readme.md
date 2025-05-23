先来简述一下项目的主要内容：

- 项目的主要内容是基于 MINIST 数据集的图像去雾算法的实现。
- 项目的主要代码文件为：

  - build_attention_layer.py: 实现了基于 MINIST 数据集的图像去雾算法的注意力机制。
    1.CrossAttention 类是一个用于处理神经网络中注意力机制的模块。它接收输入 tokens 和（可选的）上下文信息。如果用于自注意力，则专注于输入 tokens 之间的关系；在交叉注意力的情况下，考虑输入 tokens 和上下文信息之间的交互。该模块使用线性投影进行查询、键和值的转换。它计算分数矩阵、应用 softmax 得到注意力权重，并通过结合加权的值计算上下文向量。前向方法实现了这些操作，返回注意力后的上下文向量。
    2.TransformerBlock 类表示 transformer 模型中的一个构建块，结合了自注意力、交叉注意力和前馈神经网络。它接收形状为 [batch, sequence_len, hidden_dim] 的输入张量，以及（可选的）形状为 [batch, context_seq_len, context_dim] 的上下文张量。自注意力和交叉注意力模块后接层归一化和残差连接。此外，该块还包含一个具有 GELU 非线性激活函数的两层 MLP，用于进一步的非线性变换。输出是通过 TransformerBlock 后得到的张量。
    3.SpatialTransformer 层

  - Unet_Transformer.py：带有注意力层的 U-Net 架构

  - loss_function_condition.py: 训练期间加入 y 信息来更新损失函数.更新的损失函数计算带有附加条件的生成模型的损失。它包括采样时间、生成噪声、扰动输入数据，并基于模型的得分和扰动计算损失

  - randomly_gen_visua_num_attention.py: 通过注意力层添加条件生成，我们可以指示我们的稳定扩散模型绘制任何数字。让我们看看模型在绘制数字 9 时的表现

  - e_noise_forward_diffusion.py: 指数噪声的前向扩散过程

  - sampler.py
    稳定扩散通过从完全随机的图像开始创建图像。然后，噪声预测器猜测图像的噪声程度，并从图像中移除该猜测的噪声。这个循环重复多次，最终产生一个干净的图像。这种清理过程被称为“采样”，因为稳定扩散在每个步骤中都会生成一个新的图像样本。创建这些样本的方法称为“采样器”或“采样方法”。稳定扩散有多种创建图像样本的方法，我们将使用的一种方法是 Euler–Maruyama 方法，也称为 Euler 方法。此函数使用 Euler-Maruyama 方法生成图像样本，结合基于得分的模型、噪声标准差函数和扩散系数函数。它在指定的步骤数上迭代应用该方法，返回最终生成的样本集

现在简述训练过程：

- 首先，我们需要准备数据集。MINIST 数据集是一个手写数字数据集，包含 60,000 张训练图像和 10,000 张测试图像。我们将使用这些数据集来训练我们的模型。

- 然后，我们需要构建注意力层。我们将使用基于 MINIST 数据集的图像去雾算法的注意力机制。

- 接下来，我们需要构建 U-Net 架构。U-Net 架构是一种用于图像分割的深度学习模型。它由编码器和解码器组成，编码器用于提取图像特征，解码器用于重建图像。我们将使用带有注意力层的 U-Net 架构。

- 然后，我们需要定义损失函数。损失函数用于衡量模型的输出与真实值之间的差距。我们将使用带有条件的生成模型的损失函数。

- 最后，我们需要训练模型。我们将使用稳定扩散算法来训练模型。稳定扩散算法是一种用于生成图像的算法，它通过从完全随机的图像开始创建图像，然后通过噪声预测器猜测图像的噪声程度，并从图像中移除该猜测的噪声。这个循环重复多次，最终产生一个干净的图像。我们将使用 Euler-Maruyama 方法来生成图像样本，结合基于得分的模型、噪声标准差函数和扩散系数函数。我们将训练模型，使得模型能够生成数字 9。

现在，我以大白话简述一下：
一个 unet 网络结构（Unet_Tranformer.py + bulid_attention.py）
一个指数前向扩散函数（e_noise_forward_diffusion.py）
一个去噪条件的损失函数（loss_function_condition.py）
一个采样器（sampler.py）
一个时间嵌入向量函数（time_embedding_net.py）
一个训练函数(train_with_attention.py)
一个生成图像函数（randomly_gen_visua_num_attention.py）

谢谢大家！
