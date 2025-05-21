import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from thop import profile, clever_format
from ptflops import get_model_complexity_info
from .torch_wavelets import DWT_2D, IDWT_2D
from .BSRN_arch import BSConvU, CCALayer, ESDB
from .shufflemixer_arch import FMBlock
from .Bidomain import SpaBlock
from torchinfo import summary
from timm.models.layers import CondConv2d
from fvcore.nn import FlopCountAnalysis


class DynamicConv(nn.Module):
    """ Dynamic Conv layer
    """
    def __init__(self, in_features, out_features, kernel_size=1, stride=1, padding='', dilation=1,
                 groups=1, bias=False, num_experts=4):
        super().__init__()
        #print('+++', num_experts)
        self.num_experts = num_experts
        self.routing = nn.Linear(in_features, num_experts)
        self.cond_conv = CondConv2d(in_features, out_features, kernel_size, stride, padding, dilation,
                                    groups, bias, num_experts)

    def forward(self, x):
        pooled_inputs = F.adaptive_avg_pool2d(x, 1).flatten(1)  # CondConv routing
        #print(f"Pooled inputs shape: {pooled_inputs.shape}")
        routing_weights = torch.sigmoid(self.routing(pooled_inputs))
        #print(f"DynamicConv called in {self._get_parent_module()} with {self.num_experts} experts.")
        x = self.cond_conv(x, routing_weights)
        return x
    def _get_parent_module(self):
        return self.__class__.__name__

class DoubleConv(nn.Module):
    """(DynamicConv => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, num_experts=4):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            DynamicConv(in_channels, mid_channels, kernel_size=3, padding=1, num_experts=num_experts),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            DynamicConv(mid_channels, out_channels, kernel_size=3, padding=1, num_experts=num_experts),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels,num_experts=2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, num_experts=num_experts)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x1):
        return self.up(x1)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        #self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv = DynamicConv(in_channels, out_channels, kernel_size=1, num_experts=4)

    def forward(self, x):
        return self.conv(x)


class WaveBottleNeck(nn.Module):
    def __init__(self, in_ch=64, n_lo_block=1):
        super().__init__()

        self.dwt = DWT_2D(wave='haar')
        self.idwt = IDWT_2D(wave='haar')

        self.lo_blocks = nn.Sequential(
            *[FMBlock(dim=in_ch, kernel_size=7, mlp_ratio=1.25, conv_ratio=1.5) for _ in range(n_lo_block)]
        )
        #self.esdb = ESDB(in_channels=in_ch, out_channels=in_ch, conv=BSConvU)

    def forward(self, x):
        x_w = self.dwt(x)
        x_ll, x_lh, x_hl, x_hh = x_w.chunk(4, dim=1)
        x_ll = self.lo_blocks(x_ll)

        x_out = self.idwt(torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1))
        #print(x_out.shape)
        #out = self.esdb(x_out) + x
        #print(out.shape)
        return x_out + x

class BidomainNonlinearMapping(nn.Module):

    def __init__(self, in_nc):
        super(BidomainNonlinearMapping, self).__init__()
        self.spatial_process = SpaBlock(in_nc)
        self.frequency_process = WaveBottleNeck(in_nc)
        self.cat = nn.Conv2d(2 * in_nc, in_nc, 1, 1, 0)

    def forward(self, x):
        _, _, H, W = x.shape
        #x_freq = torch.fft.rfft2(x, norm='backward')
        x_spatial = self.spatial_process(x)
        x = x_spatial + x
        x_freq = self.frequency_process(x)
        #x_freq_spatial = torch.fft.irfft2(x_freq, s=(H, W), norm='backward')

        xcat = torch.cat([x, x_freq], 1)
        x_out = self.cat(xcat)

        return x_out

class AdaptiveAttentionGate(nn.Module):
    def __init__(self, encoder_dim, global_dim, num_heads=8):
        super(AdaptiveAttentionGate, self).__init__()
        self.num_heads = num_heads
        self.query_conv = nn.Conv2d(global_dim, global_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(encoder_dim, global_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(encoder_dim, global_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gate = nn.Conv2d(global_dim, 1, kernel_size=1)
        self.norm = nn.LayerNorm(global_dim)
        self.head_dim = global_dim // num_heads
        assert self.head_dim * num_heads == global_dim, "global_dim must be divisible by num_heads"

        # 如果 global_dim 不等于 encoder_dim，则需要一个额外的卷积层来匹配维度
        self.output_conv = nn.Conv2d(global_dim, encoder_dim, kernel_size=1) if global_dim != encoder_dim else None

    def forward(self, encoder_output, global_output):
        B, _, H, W = encoder_output.size()
        _, _, Hg, Wg = global_output.size()

        # 如果全局输出的空间尺寸与编码器输出的空间尺寸不匹配，则进行上采样
        if (H != Hg) or (W != Wg):
            global_output = F.interpolate(global_output, size=(H, W), mode='bilinear', align_corners=False)

        # 计算查询、键和值，并分成多头
        query = self.query_conv(global_output).view(B, self.num_heads, self.head_dim, H, W)  # (B, num_heads, head_dim, H, W)
        key = self.key_conv(encoder_output).view(B, self.num_heads, self.head_dim, H, W)    # (B, num_heads, head_dim, H, W)
        value = self.value_conv(encoder_output).view(B, self.num_heads, self.head_dim, H, W) # (B, num_heads, head_dim, H, W)

        # 展平成二维
        query_flat = query.reshape(B, self.num_heads, self.head_dim, -1)  # (B, num_heads, head_dim, H*W)
        key_flat = key.reshape(B, self.num_heads, self.head_dim, -1)      # (B, num_heads, head_dim, H*W)
        value_flat = value.reshape(B, self.num_heads, self.head_dim, -1)  # (B, num_heads, head_dim, H*W)

        # 计算注意力权重
        attention_scores = torch.einsum('bnhd,bmhd->bhnm', query_flat, key_flat)  # (B, num_heads, H*W, H*W)
        attention_weights = self.softmax(attention_scores)  # (B, num_heads, H*W, H*W)

        # 计算加权求和值
        attention_output = torch.einsum('bhnm,bmhd->bnhd', attention_weights, value_flat)  # (B, num_heads, head_dim, H*W)
        attention_output = attention_output.reshape(B, -1, H, W)  # (B, global_dim, H, W)

        attention_output = attention_output + global_output

        # 对注意力输出应用 LayerNorm
        attention_output = self.norm(attention_output.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # 计算自适应注意力门的权重
        gate_weight = torch.sigmoid(self.gate(attention_output))  # (B, 1, H, W)

        # 调节信息流的重要程度
        gated_output = attention_output * gate_weight  # (B, global_dim, H, W)

        # 如果需要，将 global_dim 转换为 encoder_dim
        if self.output_conv is not None:
            gated_output = self.output_conv(gated_output)

        # 进行残差连接
        output = gated_output + encoder_output  # 残差连接

        return output

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    padding = int((kernel_size - 1) / 2) * dilation
    depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding=padding, bias=bias, dilation=dilation, groups=in_channels)
    pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
    return nn.Sequential(depthwise_conv, pointwise_conv)

def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


class IMDModule(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(IMDModule, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = int(in_channels - self.distilled_channels)
        self.c1 = conv_layer(in_channels, in_channels, 3, groups=2)
        self.c2 = conv_layer(self.remaining_channels, in_channels, 3, groups=2)
        self.c3 = conv_layer(self.remaining_channels, in_channels, 3, groups=2)
        self.c4 = conv_layer(self.remaining_channels, self.distilled_channels, 3, groups=2)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(in_channels, in_channels, 1, groups=2)
        self.cca = CCALayer(self.distilled_channels * 4)

    def forward(self, input):
        out_c1 = self.act(self.c1(input))
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c2 = self.act(self.c2(remaining_c1))
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c3 = self.act(self.c3(remaining_c2))
        distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c4 = self.c4(remaining_c3)
        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4], dim=1)
        out_fused = self.c5(self.cca(out)) + input
        return out_fused

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        # Global average pooling
        y = self.avg_pool(x)
        # Pass through fully connected layers
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        #y = self.dropout1(y)
        return y * x  # Element-wise multiplication

class GlobalInformationAggregate(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GlobalInformationAggregate, self).__init__()
        self.channelFusion = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.imd_module = IMDModule(out_channels)

        # 定义3个不同大小的深度可分离卷积
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)  # Pointwise卷积
        )
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2, groups=out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)  # Pointwise卷积
        )
        self.conv7x7 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=7, padding=3, groups=out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)  # Pointwise卷积
        )

        # SE Blocks
        self.se_block3 = SEBlock(out_channels)
        self.se_block5 = SEBlock(out_channels)
        #self.se_block7 = SEBlock(out_channels)

    def forward(self, x):
        x_1, x_2, x_3, x_4, x_5 = x
        B, C, H, W = x_4.shape
        output_size = (H, W)

        # 动态调整池化大小
        x_1 = F.adaptive_avg_pool2d(x_1, output_size)
        x_2 = F.adaptive_avg_pool2d(x_2, output_size)
        x_3 = F.adaptive_avg_pool2d(x_3, output_size)
        x_5 = F.interpolate(x_5, size=(H, W), mode='bilinear', align_corners=False)

        # 拼接并进行通道融合
        out = torch.cat([x_1, x_2, x_3, x_4, x_5], dim=1)
        out_1 = self.channelFusion(out)

        ### 分别通过3个深度可分离卷积层
        out_3x3 = self.conv3x3(out_1)  # 通过3x3深度可分离卷积
        out_5x5 = self.conv5x5(out_1)  # 通过5x5深度可分离卷积
        out_7x7 = self.conv7x7(out_1)  # 通过7x7深度可分离卷积

        ### 依次通过3x3 -> 5x5 -> 7x7深度可分离卷积
        out_seq_3x5x7 = self.conv3x3(out_1)
        out_seq_3x5x7 = self.conv5x5(out_seq_3x5x7)
        out_seq_3x5x7 = self.conv7x7(out_seq_3x5x7)

        # 将 3x3 和 5x5 的输出相加，将 7x7 的输出相加，再加上依次经过的结果
        x35 = out_3x3 + out_5x5
        x7357 = out_7x7 + out_seq_3x5x7

        # 通过 SE Block 增强
        x35 = self.se_block3(x35)
        x7357 = self.se_block5(x7357)

        # 元素逐乘
        x3 = x35 * out_3x3
        x5 = x35 * out_5x5
        x7 = x7357 * out_7x7
        x_seq = x7357 * out_seq_3x5x7

        # 最终输出融合
        x_fused = x3 + x5 + x7 + x_seq

        # 通过IMD模块并加回out_1
        out_imd = self.imd_module(x_fused)
        out = out_imd + out_1

        return out

def depthwise_separable_conv(in_channels, out_channels, kernel_size=3, padding=1):
    return nn.Sequential(
        # 深度卷积
        nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=False),
        # 逐点卷积
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    )

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = (DoubleConv(n_channels, 64, num_experts=2))
        self.down1 = (Down(64, 128,num_experts=2))
        self.down2 = (Down(128, 256,num_experts=2))
        self.down3 = (Down(256, 512,num_experts=4))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor, num_experts=4))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

        #self.attention1 = AdaptiveAttentionGate(512, 256)
        #self.attention2 = AdaptiveAttentionGate(256, 256)
        #self.attention3 = AdaptiveAttentionGate(128, 256)
        #self.attention4 = AdaptiveAttentionGate(64, 256)

        #self.global_info = GlobalInformationAggregate(1472, 256)

        self.conv1 = depthwise_separable_conv(512, 256, kernel_size=3, padding=1)
        self.conv2 = depthwise_separable_conv(256, 128, kernel_size=3, padding=1)
        self.conv3 = depthwise_separable_conv(128, 64, kernel_size=3, padding=1)
        self.conv4 = depthwise_separable_conv(64, 64, kernel_size=3, padding=1)

        self.wavebottleneck1 = WaveBottleNeck(in_ch=64)
        self.wavebottleneck2 = WaveBottleNeck(in_ch=128)
        self.wavebottleneck3 = WaveBottleNeck(in_ch=256)
        self.wavebottleneck4 = WaveBottleNeck(in_ch=512)
        self.wavebottleneck5 = WaveBottleNeck(in_ch=512)

        self.bnm1 = BidomainNonlinearMapping(64)
        self.bnm2 = BidomainNonlinearMapping(128)
        self.bnm3 = BidomainNonlinearMapping(256)
        self.bnm4 = BidomainNonlinearMapping(512)
        self.bnm5 = BidomainNonlinearMapping(512)

    def forward(self, x):
        x1 = self.inc(x)# Extract features from the original image (x1)
        x1_freq = self.wavebottleneck1(x1)  # Dehaze x1 to get x1_dehaze
        x1_dehaze = self.bnm1(x1_freq)

        x2 = self.down1(x1_dehaze)  # Downsample x1_dehaze to get x2
        x2_freq = self.wavebottleneck2(x2)  # Dehaze x2 to get x2_dehaze
        x2_dehaze = self.bnm2(x2_freq)

        x3 = self.down2(x2_dehaze)  # Downsample x2_dehaze to get x3
        x3_freq = self.wavebottleneck3(x3)  # Dehaze x3 to get x3_dehaze
        x3_dehaze = self.bnm3(x3_freq)

        x4 = self.down3(x3_dehaze)  # Downsample x3_dehaze to get x4
        x4_freq = self.wavebottleneck4(x4)  # Dehaze x4 to get x4_dehaze
        x4_dehaze = self.bnm4(x4_freq)

        x5 = self.down4(x4_dehaze)  # Downsample x4_dehaze to get x5
        x5_freq = self.wavebottleneck5(x5)  # Dehaze x5 to get x5_dehaze
        x5_dehaze = self.bnm5(x5_freq)

        #global_output = self.global_info((x1_dehaze, x2_dehaze, x3_dehaze, x4_dehaze, x5_dehaze))  # Global information aggregation
        #print(x1.shape, x2.shape, x3.shape, x4.shape, x5_dehaze.shape, global_output.size())

        d1 = self.up1(x5_dehaze)
        #x4_fusion = self.attention1(x4_dehaze, global_output)
        d1 = x4_dehaze + d1
        d1 = self.conv1(d1)

        d2 = self.up2(d1)
        #x3_fusion = self.attention2(x3_dehaze, global_output)
        d2 = x3_dehaze + d2
        d2 = self.conv2(d2)

        d3 = self.up3(d2)
        #x2_fusion = self.attention3(x2_dehaze, global_output)
        d3 = x2_dehaze + d3
        d3 = self.conv3(d3)

        d4 = self.up4(d3)
        #x1_fusion = self.attention4(x1_dehaze, global_output)
        d4 = x1_dehaze + d4
        d4 = self.conv4(d4)

        logits = self.outc(d4)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


if __name__ == '__main__':
    model = UNet(n_channels=3, n_classes=2).cuda()
    input_tensor = torch.rand([1, 3, 256, 256]).cuda()
    
    # 使用 get_model_complexity_info 计算 mac 和参数数量
    mac_info, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True, print_per_layer_stat=False)
    
    # 使用 FlopCountAnalysis 计算 FLOPs
    flops_analysis = FlopCountAnalysis(model, input_tensor)
    
    # 打印结果
    print(f"Output from get_model_complexity_info: MACs = {mac_info}, Params = {params}")
    print(f"Output from FlopCountAnalysis: FLOPs = {flops_analysis.total()/1e9:.2f}")





    #Print model summary
    #summary(model, input_size=(1, 3, 256, 256))