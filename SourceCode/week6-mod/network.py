import torch
import torch.nn as nn
import thop

class ConvBlock(nn.Module):
    def __init__(self, in_feat, out_feat, k=3, s=1, p=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_feat, out_feat, k, s, p, bias=False),
            nn.BatchNorm2d(out_feat),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.body(x)
        return x

class modelA(nn.Module):
    def __init__(self):
        super(modelA, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(3, 15),
            ConvBlock(15, 30),
            ConvBlock(30, 60),
            ConvBlock(60, 128),
            ConvBlock(128, 256)
        )
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.head = nn.Sequential(nn.Linear(256, 10))

    def forward(self, x):
        x = self.block(x)
        x = self.pool(x).squeeze()
        x = self.head(x)
        return x
    
class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(DepthwiseConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.depthwise(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class PointwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PointwiseConv, self).__init__()
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.pointwise(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class modelB(nn.Module):
    def __init__(self):
        super(modelA, self).__init__()
        self.block = nn.Sequential(
            DepthwiseConv(3, 3),
            PointwiseConv(3, 30),

            DepthwiseConv(30, 3),
            PointwiseConv(30, 64),

            DepthwiseConv(64, 3),
            PointwiseConv(64, 100),

            DepthwiseConv(100, 3),
            PointwiseConv(100, 120),

            DepthwiseConv(120, 3),
            PointwiseConv(120, 128)
        )

        self.pool = nn.AdaptiveMaxPool2d(1)
        self.head = nn.Linear(128, 10)

    def forward(self, x):
        x = self.block(x)
        x = self.pool(x).squeeze()
        x = self.head(x)
        return x
    
if __name__ == "__main__":
    pass