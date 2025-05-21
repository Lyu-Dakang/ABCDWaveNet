import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#from torch.nn.init import _calculate_fan_in_and_fan_out_

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size ** 2, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def get_relative_positions(window_size):
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)

    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_positions = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww

    relative_positions = relative_positions.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_positions_log = torch.sign(relative_positions) * torch.log(1. + relative_positions.abs())

    return relative_positions_log

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        relative_positions = get_relative_positions(self.window_size)
        self.register_buffer("relative_positions", relative_positions)
        self.meta = nn.Sequential(
            nn.Linear(2, 256, bias=True),
            nn.ReLU(True),
            nn.Linear(256, num_heads, bias=True)
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv):
        B_, N, _ = qkv.shape

        qkv = qkv.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.meta(self.relative_positions)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, self.dim)
        return x
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, relu=True):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU(inplace=True) if relu else nn.Identity()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ResBlock_Conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock_Conv, self).__init__()
        self.conv1 = DepthwiseSeparableConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True)
        self.conv2 = DepthwiseSeparableConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        self.trans_layer = DepthwiseSeparableConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans_layer(out)
        out = self.conv2(out)
        return out + x

class SpaBlock(nn.Module):
    def __init__(self, nc):
        super(SpaBlock, self).__init__()
        self.block = ResBlock_Conv(in_channel=nc, out_channel=nc)

    def forward(self, x):
        yy = self.block(x)
        return yy

