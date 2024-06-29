import torch
import torch.nn as nn
import torch.nn.functional as F


class AvgSpatial(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, x.size()[2:])


class TripleConv(nn.Module):
    def __init__(self, num_channels):
        super(TripleConv, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(num_channels, num_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(num_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        do_bn = x.size(2) > 1 and x.size(3) > 1 # Check if spatial dimensions are larger than 1x1
        identity = x

        out = self.conv1(x)
        if do_bn:
            out = self.bn1(out)
        out = self.relu1(out)

        out += identity  # Skip connection

        out = self.conv2(out)
        if do_bn:  # Check again for spatial dimensions
            out = self.bn2(out)
        out = self.relu2(out)

        out += identity  # Another skip connection

        out = self.conv3(out)
        if do_bn:  # Final check for spatial dimensions
            out = self.bn3(out)
        out = self.relu3(out)

        return out


def max_channel_pool(x):
    return x.max(dim=1, keepdim=True)[0]


def avg_channel_pool(x):
    return x.mean(dim=1, keepdim=True)


class ChannelAttention(nn.Module):
    def __init__(self, num_channels):
        super(ChannelAttention, self).__init__()
        self.avg_spatial = AvgSpatial()  # Average spatial pooling
        self.triple_conv = TripleConv(num_channels)  # Triple convolution block
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation

    def forward(self, x):
        out = self.avg_spatial(x)  # Eq. (8)
        out = self.triple_conv(out)  # Eq. (9)
        out = self.sigmoid(out) * x  # Eq. (10)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, num_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)  # Convolution for spatial attention
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation

    def forward(self, x):
        c1 = max_channel_pool(x)  # Eq. (11)
        c2 = avg_channel_pool(x)  # Eq. (12)
        c3 = torch.cat([c1, c2], dim=1)  # Eq. (13)
        c4 = self.conv(c3)  # Eq. (14)
        c5 = self.sigmoid(c4) * x  # Eq. (15)
        return c5


class MultimodalFusion(nn.Module):
    def __init__(self, num_channels):
        super(MultimodalFusion, self).__init__()
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

    def forward(self, ti, ci):
        # Adjusting to match the paper's description -
        # early version contained a mistake (concatenation instead of Hadamard)
        hadamard_product = ti * ci  # Element-wise multiplication (Hadamard product)
        fused = self.conv(hadamard_product)  # Apply convolution
        return fused


class BiFusionBlock(nn.Module):
    def __init__(self, num_channels):
        super(BiFusionBlock, self).__init__()
        self.ca = ChannelAttention(num_channels)  # Channel Attention
        self.sa = SpatialAttention(num_channels)  # Spatial Attention
        self.fusion = MultimodalFusion(num_channels)  # Multimodal Fusion
        self.residual = nn.Sequential(
            nn.Conv2d(num_channels * 3, num_channels, kernel_size=1),  # Residual connection
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, ti, ci):
        t3 = self.ca(ti)  # Output of Channel Attention
        c5 = self.sa(ci)  # Output of Spatial Attention
        fi = self.fusion(ti, ci)  # Output of Multimodal Fusion
        concat = torch.cat([t3, c5, fi], dim=1)  # Concatenate outputs
        out = self.residual(concat)  # Pass through residual module
        return out


# some tests
if __name__ == '__main__':
    import torch.optim as optim

    class SimpleModel(nn.Module):
        def __init__(self, num_channels):
            super(SimpleModel, self).__init__()
            self.bifusion = BiFusionBlock(num_channels)

        def forward(self, ti, ci):
            return self.bifusion(ti, ci)


    # Example model, loss function, optimizer, and input batch
    model = SimpleModel(16)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    batch_ti = torch.randn(1, 16, 64, 64)  # Example transformer branch input
    batch_ci = torch.randn(1, 16, 64, 64)  # Example CNN branch input


    def test_inf_values():
        inf_input = torch.tensor(float('inf')) * torch.ones_like(batch_ti)
        try:
            output = model(inf_input, batch_ci)
            print("Test Inf Values Passed")
        except Exception as e:
            print(f"Test Inf Values Failed: {e}")

    def test_nan_values():
        nan_input = torch.tensor(float('nan')) * torch.ones_like(batch_ti)
        try:
            output = model(nan_input, batch_ci)
            print("Test NaN Values Passed")
        except Exception as e:
            print(f"Test NaN Values Failed: {e}")

    def test_output_shape():
        output = model(batch_ti, batch_ci)
        expected_shape = (1, 16, 64, 64)  # Assuming no change in spatial dimensions and channels
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
        print("Test Output Shape Passed")

    test_inf_values()
    test_nan_values()
    test_output_shape()
