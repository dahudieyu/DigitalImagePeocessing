import cv2
import numpy as np
 
def dark_channel_prior(image, kernel_size=15):
    # 分割图像通道
    b, g, r = cv2.split(image)
    # 计算三个通道中的最小值
    min_channel = cv2.min(cv2.min(r, g), b)
    # 使用指定大小的矩形卷积核进行腐蚀操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dark_channel = cv2.erode(min_channel, kernel)
    return dark_channel
 
# 读取图像
image = cv2.imread("2.png")
 
# 计算暗通道
dark_channel = dark_channel_prior(image)
 
# 显示结果
cv2.imshow('Dark Channel', dark_channel)
cv2.waitKey(0)
cv2.destroyAllWindows()