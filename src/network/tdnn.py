import torch
from torch import nn


class ECAPA_TDNN(nn.Module):
    """
    ECAPA-TDNN模型结构
    [B, D, T] -> [B, E]
    """
    def __init__(self, input_dim: int, embedding_size: int, mid_channels: int=512):
        super(ECAPA_TDNN, self).__init__()
        
        self.conv1 = ConvReluBn(input_dim, mid_channels, kernel_size=5, dilation=1, padding=2)
        self.seres2block1 = SERes2Block(mid_channels, res2_scale=4, se_scale=4, split_num=8, kernel_size=3, dilation=2)
        self.seres2block2 = SERes2Block(mid_channels, res2_scale=4, se_scale=4, split_num=8, kernel_size=3, dilation=3)
        self.seres2block3 = SERes2Block(mid_channels, res2_scale=4, se_scale=4, split_num=8, kernel_size=3, dilation=4)
        self.conv2 = nn.Conv1d(mid_channels * 3, mid_channels * 3, kernel_size=1)
        self.relu = nn.ReLU()
        self.attentive_stats_pool = AttentiveStatsPool(mid_channels * 3, scale=4)
        self.fc = nn.Linear(mid_channels * 6, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x1 = self.seres2block1(x)
        x2 = self.seres2block2(x1)
        x3 = self.seres2block3(x2)
        out = torch.cat([x1, x2, x3], dim=1)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.attentive_stats_pool(out)
        out = self.fc(out)
        out = self.bn(out)

        return out


class SERes2Block(nn.Module):
    """
    SE-Res2Block模块
    [B, C_in, T] -> [B, C_out, T]
    """
    def __init__(self, channels: int, res2_scale: int, se_scale: int, split_num: int, kernel_size: int, dilation: int):
        super(SERes2Block, self).__init__()
        self.layers = nn.Sequential(
            Res2Block(channels, res2_scale, kernel_size, dilation, split_num),
            SEBlock(channels, se_scale)
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class ConvReluBn(nn.Module):
    """
    基本卷积单元
    [B, C_in, T] -> [B, C_out, T]
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, padding: int):
        super(ConvReluBn, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)
    

class Res2Block(nn.Module):
    """
    残差连接模块
    [B, C_in, T] -> [B, C_out, T]
    """
    def __init__(self, channels: int, scale: int, kernel_size: int, dilation: int, split_num: int):
        super(Res2Block, self).__init__()
        assert channels % scale == 0, f"channels must be divisible by scale, but got channels={channels} and scale={scale}"
        assert channels // scale % split_num == 0, f"sub_channels must be divisible by split_num, but got sub_channels={channels // scale} and split_num={split_num}"

        self.conv1 = ConvReluBn(channels, channels // scale, kernel_size=1, dilation=1, padding=0)

        self.sub_channels = channels // scale // split_num
        self.scale = scale
        self.split_num = split_num
        self.sub_convs = nn.ModuleList(
            [
                ConvReluBn(self.sub_channels, self.sub_channels, kernel_size, dilation=dilation, padding=(kernel_size-1)*dilation//2) for _ in range(self.split_num - 1)
            ]
        )

        self.conv2 = ConvReluBn(channels // scale, channels, kernel_size=1, dilation=1, padding=0)

    def forward(self, x: torch.Tensor):
        out = self.conv1(x)
        sub_outs = torch.split(out, self.sub_channels, dim=1)
        final_outs = [sub_outs[0]]
        for i in range(self.split_num - 1):
            final_outs.append(self.sub_convs[i](sub_outs[i + 1] + final_outs[i]))
        out = torch.cat(final_outs, dim=1)
        out = self.conv2(out)

        return out + x
    

class SEBlock(nn.Module):
    """
    SE注意力机制模块
    [B, C, T] -> [B, C, T]
    """
    def __init__(self, channels: int, scale: int):
        super(SEBlock, self).__init__()
        assert channels % scale == 0, f"channels must be divisible by scale, but got channels={channels} and scale={scale}"

        self.weight_layers = nn.Sequential(
            nn.Linear(channels, channels // scale),
            nn.ReLU(),
            nn.Linear(channels // scale, channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        w = torch.mean(x, dim=2)  # [B, C]
        w = self.weight_layers(w)  # [B, C]
        w = w.unsqueeze(2)  # [B, C, 1]
        return x * w  # [B, C, T]
    

class AttentiveStatsPool(nn.Module):
    """
    注意力统计池化模块
    [B, C, T] -> [B, 2C]
    """
    def __init__(self, channels: int, scale: int):
        super(AttentiveStatsPool, self).__init__()
        assert channels % scale == 0, f"channels must be divisible by scale, but got channels={channels} and scale={scale}"
        self.attention_layers = nn.Sequential(
            nn.Conv1d(channels, channels // scale, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(channels // scale, 1, kernel_size=1),
            nn.Softmax(dim=2)
        )

    def forward(self, x: torch.Tensor):
        alpha = self.attention_layers(x)  # [B, 1, T]
        mean = torch.sum(alpha * x, dim=2)  # [B, C]
        std = torch.sqrt(torch.sum(alpha * x ** 2, dim=2) - mean ** 2 + 1e-9)  # [B, C]
        return torch.cat([mean, std], dim=1)  # [B, 2C]
