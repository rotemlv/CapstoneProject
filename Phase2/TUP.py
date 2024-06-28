import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO:
"""
Rotem:
The implementation details leave much information to be desired. I am assuming ReLU is used after each conv. block,
this is not specified in the original paper.
Additionally, I assume the "X" in the paper means "multiply" as stated in a different figure, and that the
multiplication operation is element-wise (the only way dimensions make sense).
Unfortunately, with no insight from the original authors, we are left with deducing with what we have.

This block is subject to extensive modification after thorough testing.     
"""


class StdConv2d(nn.Conv2d):
    """This class is taken from the TransUNet project, testing required to check if the normalization here makes the
    BN layers unnecessary."""

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)  # (avoid dividing by 0)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    """A wrapper for a single StdConv3x3 block"""
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def ResidualPath(cin, cout, stride=1, bias=False):
    """A wrapper for the residual path's operations (poorly written for now)"""
    return nn.Sequential(StdConv2d(cin, cout, kernel_size=1, stride=stride,
                                   padding=0, bias=bias), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2
    Currently uses the wrapper above"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            conv3x3(in_channels, mid_channels),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            conv3x3(mid_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class TUP(nn.Module):
    """Transformer UP-sampler.
    Inputs: transformer output features
    Outputs: transformer output features scaled up, in preparation to enter BiFusion block.
    """

    def __init__(self, cin, cout):
        super().__init__()
        cmid = (cin + cout) // 2
        # upsample
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        # two 3x3 conv -> batch norms
        self.dblConv = DoubleConv(cin, cout, cmid)
        # residual path
        self.conv1 = ResidualPath(cin, cout)

    def forward(self, x):
        x = self.up(x)

        # Residual branch
        residual = x
        residual = self.conv1(residual)

        # Main branch
        y = self.dblConv(x)
        y *= residual
        return y


# Well this part is copied:
if __name__ == '__main__':
    # Instantiate the model with example input/output channels
    model = TUP(cin=16, cout=32)

    # Create dummy input data
    # Assuming the input is a 4D tensor (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 16, 64, 64)  # Example: Batch size of 1, 16 input channels, 64x64 spatial dimensions

    # Forward pass through the model
    output = model(dummy_input)

    # Print the output shape to verify the model's functionality
    print("Output shape:", output.shape)
