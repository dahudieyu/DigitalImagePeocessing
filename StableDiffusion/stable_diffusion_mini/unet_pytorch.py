# # pytorch 实现的 U-Net 网络
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels=3):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, input_channels=3, text_embed_dim=512, feature_map_size=64, bilinear=True):
        super(UNet, self).__init__()

        # 文本条件化
        self.text_conditioning = nn.Linear(text_embed_dim, feature_map_size * 2)

        self.inc = DoubleConv(input_channels, feature_map_size)
        self.down1 = Down(feature_map_size, feature_map_size * 2)
        self.down2 = Down(feature_map_size * 2, feature_map_size * 4)
        self.down3 = Down(feature_map_size * 4, feature_map_size * 8)
        self.down4 = Down(feature_map_size * 8, feature_map_size * 16)
        self.up1 = Up(feature_map_size * 16, feature_map_size * 8, bilinear)
        self.up2 = Up(feature_map_size * 8, feature_map_size * 4, bilinear)
        self.up3 = Up(feature_map_size * 4, feature_map_size * 2, bilinear)
        self.up4 = Up(feature_map_size * 2, feature_map_size, bilinear)
        self.outc = OutConv(feature_map_size, input_channels)

    def forward(self, x, text_embedding):
        """
        x: 输入噪声图像，形状 (batch_size, input_channels, height, width)
        text_embedding: 文本嵌入，形状 (batch_size, text_embed_dim)
        """
        # 文本条件化
        text_emb = self.text_conditioning(text_embedding).unsqueeze(-1).unsqueeze(-1)  # (batch_size, feature_map_size*2, 1, 1)
        text_emb = text_emb.expand(-1, -1, x.size(2), x.size(3))  # 扩展到图像大小 

        # 编码器部分
        x = x + text_emb  # 将文本嵌入信息与输入图像相结合
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 解码器部分
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == '__main__':
    # 示例: 输入图像为3通道，文本嵌入维度为512，输出为1通道（用于图像生成）
    net = UNet(input_channels=3, text_embed_dim=512)
    print(net)








# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class UNet(nn.Module):
#     #def __init__(self, input_channels=3, text_embed_dim=768, 、feature_map
#     def __init__(self, input_channels=3, text_embed_dim=512, feature_map_size=64):
#         super().__init__()
        
#         # 编码器（Encoder）部分
#         self.enc1 = self.conv_block(input_channels, feature_map_size)
#         self.enc2 = self.conv_block(feature_map_size, feature_map_size*2)
#         self.enc3 = self.conv_block(feature_map_size*2, feature_map_size*4)
#         self.enc4 = self.conv_block(feature_map_size*4, feature_map_size*8)
        
#         # 底部（Bottleneck）
#         self.bottleneck = self.conv_block(feature_map_size*8, feature_map_size*16)
        
#         # 解码器（Decoder）部分
#         self.dec4 = self.deconv_block(feature_map_size*16, feature_map_size*8)
#         self.dec3 = self.deconv_block(feature_map_size*8, feature_map_size*4)
#         self.dec2 = self.deconv_block(feature_map_size*4, feature_map_size*2)
#         self.dec1 = self.deconv_block(feature_map_size*2, feature_map_size)
        
#         # 最后一层，输出通道为 3（RGB 图像）
#         self.final_conv = nn.Conv2d(feature_map_size, input_channels, kernel_size=1)
        
#         # 文本嵌入（用于条件化）
#         self.text_conditioning = nn.Linear(text_embed_dim, feature_map_size*2)

#     def conv_block(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def deconv_block(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x, text_embedding):
#         """
#         x: 输入噪声图像，形状 (batch_size, input_channels, height, width)
#         text_embedding: 文本嵌入，形状 (batch_size, text_embed_dim)
#         """
#         # 文本条件化
#         text_emb = self.text_conditioning(text_embedding).unsqueeze(-1).unsqueeze(-1)  # (batch_size, feature_map_size*2, 1, 1)
#         text_emb = text_emb.expand(-1, -1, x.size(2), x.size(3))  # 扩展到图像大小

#         # 编码器部分
#         enc1 = self.enc1(x)
#         enc2 = self.enc2(enc1)
#         enc3 = self.enc3(enc2)
#         enc4 = self.enc4(enc3)

#         # 底部
#         bottleneck = self.bottleneck(enc4)

#         # 解码器部分
#         dec4 = self.dec4(bottleneck)
#         dec4 = dec4 + enc4  # 跳跃连接
#         dec3 = self.dec3(dec4)
#         dec3 = dec3 + enc3
#         dec2 = self.dec2(dec3)
#         dec2 = dec2 + enc2
#         dec1 = self.dec1(dec2)
#         dec1 = dec1 + enc1
        
#         # 最后一层
#         out = self.final_conv(dec1)
        
#         return out

