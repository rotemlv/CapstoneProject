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
    """Standard Convolution with weight normalization."""
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


class DoubleConv(nn.Module):
    """(convolution(3x3, pad=1) => [BN] => ReLU) * 2
    Currently uses the wrapper above"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            StdConv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            StdConv2d(mid_channels, out_channels, kernel_size=3, padding=1),
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
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dblConv = DoubleConv(cin, cout, cmid)
        self.conv1 = nn.Sequential(
            StdConv2d(cin, cout, kernel_size=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Upsample features
        upsampled_x = self.up(x)

        # Residual branch processing
        residual = self.conv1(upsampled_x)

        # Main branch processing
        y = self.dblConv(upsampled_x)

        # Integrate features through element-wise multiplication
        y *= residual

        return y


# Well this part is copied:
if __name__ == '__main__':
    # Instantiate the model with example input/output channels
    model = TUP(cin=16, cout=32)

    # Create dummy input data
    # Assuming the input is a 4D tensor (batch_size, channels, height, width)
    dummy_input = torch.abs(torch.randn(1, 16, 64, 64))  # Example: Batch size of 1, 16 input channels, 64x64 spatial
    # dimensions
    # print(dummy_input[0,1,:20,:20])

    # Forward pass through the model
    output = model(dummy_input)

    # Print the output shape to verify the model's functionality
    print("Output shape:", output.shape)
    # print(output[0,1,:20, :20])


"""
# old unused code:

def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    #A wrapper for a single StdConv3x3 block
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def ResidualPath(cin, cout, stride=1, bias=False):
    #A wrapper for the residual path's operations (poorly written for now)
    return nn.Sequential(StdConv2d(cin, cout, kernel_size=1, stride=stride,
                                   padding=0, bias=bias), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

"""