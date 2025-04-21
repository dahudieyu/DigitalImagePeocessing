架构图：
 文字 prompt → [CLIP] → 文本特征 → → [UNet] + 时间步t + 噪声图像 → → 预测噪声 ε → → Diffusion 模型去噪 → 生成图像


步骤总览：
步骤 1：准备数据（图像+文字）
步骤 2：CLIP 文本编码器
步骤 3：UNet 模型（+Cross-Attention）
步骤 4：Diffusion 前向加噪 & 反向训练
步骤 5：采样生成图片 