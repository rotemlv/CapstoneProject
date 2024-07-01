import torch
import torch.nn as nn
import torch.nn.functional as F

"""
This variant is loyal to the implementation mentioned in the sources provided in BiFTransNet
"""


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelAttention(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.input_channels = input_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        #  uses Convolutions instead of Linear
        self.MLP = nn.Sequential(
            Flatten(),
            nn.Linear(input_channels, input_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(input_channels // reduction_ratio, input_channels)
        )

    def forward(self, x):
        # Take the input and apply average and max pooling
        avg_values = self.avg_pool(x)
        max_values = self.max_pool(x)
        out = self.MLP(avg_values) + self.MLP(max_values)
        scale = x * torch.sigmoid(out).unsqueeze(2).unsqueeze(3).expand_as(x)
        return scale


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.bn(out)
        scale = x * torch.sigmoid(out)
        return scale


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
        self.sa = SpatialAttention()  # Spatial Attention
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
        print(f"Test Output Shape Passed, Input shape=2*{batch_ti.shape}, output shape: {output.shape}")

    test_inf_values()
    test_nan_values()
    test_output_shape()
