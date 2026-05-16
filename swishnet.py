import torch
from torch import nn

class SwishNet(nn.Module):
    def __init__(self, in_channels=20, out_channels=2, width_multiply=1):
        super().__init__()
        width_8 = 8 * width_multiply
        width_16 = 16 * width_multiply
        width_32 = 32 * width_multiply
        width_40 = 40 * width_multiply

        self.causal_block1 = CausalBlock(in_channels, width_16)
        self.causal_block2 = CausalBlock(width_16, width_8)
        self.causal_block3 = CausalBlock(width_8, width_8)
        self.causal_conv4 = CausalGatedConv1D(width_8, width_16, length=3, dilation=3)
        self.causal_conv5 = CausalGatedConv1D(width_8, width_16, length=3, dilation=2)
        self.causal_conv6 = CausalGatedConv1D(width_8, width_16, length=3, dilation=2)
        self.causal_conv7 = CausalGatedConv1D(width_8, width_16, length=3, dilation=2)
        self.causal_conv8 = CausalGatedConv1D(width_8, width_32, length=3, dilation=2)
        self.conv_out = nn.Conv1d(width_40, out_channels, kernel_size=1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x) -> torch.Tensor:
        # block 1
        x = self.causal_block1(x)
        
        # block 2
        x = self.causal_block2(x)
        
        # block 3
        x_causal = self.causal_block3(x)
        
        x = x + x_causal
        
        # block 4
        x_block4 = self.causal_conv4(x)
        x = x + x_block4
        
        # block 5
        x_block5 = self.causal_conv5(x)
        x = x + x_block5
        
        # block 6
        x_block6 = self.causal_conv6(x)
        x = x + x_block6
        
        # block 7
        x_block7 = self.causal_conv7(x)
        
        # block 8
        x_block8 = self.causal_conv8(x)
        
        x = torch.cat((x_block5, x_block6, x_block7, x_block8), dim=1)

        # output
        x = self.conv_out(x)
        x = self.global_avg_pool(x)
        x = x.squeeze(-1)

        return x

class SwishNetWide(SwishNet):
    def __init__(self, classes):
        super().__init__(out_channels=classes, width_multiply=2)

# NOTE: Copied from https://github.com/pytorch/pytorch/issues/1333#issuecomment-400338207
class CausalConv1D(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1D, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input):
        result = super(CausalConv1D, self).forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result

class CausalGatedConv1D(nn.Module):
    def __init__(self, in_channels, filters=16, length=6, dilation=1):
        super().__init__()
        self.conv1 = CausalConv1D(in_channels=in_channels, out_channels=filters // 2, kernel_size=length, dilation=dilation, stride=1)
        self.conv2 = CausalConv1D(in_channels=in_channels, out_channels=filters // 2, kernel_size=length, dilation=dilation, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x_conv1 = self.conv1(x)
        x_sigmoid = self.sigmoid(x_conv1)
        x_conv2 = self.conv2(x)
        x_tanh = self.tanh(x_conv2)
        x_out = torch.mul(x_sigmoid, x_tanh)
        return x_out

class CausalBlock(nn.Module):
    def __init__(self, in_channels, filters):
        super().__init__()
        self.causal_conv_up = CausalGatedConv1D(in_channels, filters, length=3)
        self.causal_conv_down = CausalGatedConv1D(in_channels, filters, length=6)

    def forward(self, x):
        x_up = self.causal_conv_up(x)
        x_down = self.causal_conv_down(x)
        x_out = torch.cat([x_up, x_down], dim=1) 
        return x_out

