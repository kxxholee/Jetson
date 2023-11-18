import torch
import torch.nn as nn
import thop

class ConvBNAct(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, stride, padding):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_feat, out_feat, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_feat),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.body(x)
        return x

class modelA(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(3, 15, 3, 1, 1),
            nn.Conv2d(15, 30, 3, 1, 1),
            nn.Conv2d(30, 60, 3, 1, 1),
            nn.Conv2d(60, 128, 3, 1, 1),
            nn.Conv2d(128, 256, 3, 1, 1)
        )
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.head = nn.Sequential(nn.Linear(256, 10))
    
    def forward(self, x):
        x = self.block(x)
        x = self.pool(x).squeeze()
        x = self.head(x)
        return x

# Depthwise Convolution 구현
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

# Pointwise Convolution 구현
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
        super().__init__()
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
        self.head = nn.Sequential(nn.Linear(128, 10))
    
    def forward(self, x):
        x = self.block(x)
        x = self.pool(x).squeeze()
        x = self.head(x)
        return x
    
if __name__ == "__main__":
    # 테스트를 위한 입력 데이터 생성
    input = torch.randn(1, 3, 224, 224)

    # 일반적인 컨볼루션 레이어 생성
    conv_layer = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU())
    out = conv_layer(input)

    # Depthwise convolution
    model_A = modelA()

    # Pointwise convolution
    model_B = modelB()

    # 연산량 및 파라미터 계산
    flops_conv, params_conv             = thop.profile(conv_layer, inputs=(input, ))
    flops_depthwise, params_depthwise   = thop.profile(model_A, inputs=(input, ))
    flops_pointwise, params_pointwise   = thop.profile(model_B, inputs=(input, ))

    print("Convolution (GFLOPs)         :", flops_conv)
    print("Convolution params (Millions):", params_conv)
    print("Model A (GFLOPs)             :", flops_depthwise)
    print("Model A params (Millions)    :", params_depthwise)
    print("Model B (GFLOPs)             :", flops_pointwise)
    print("Model B params (Millions)    :", params_pointwise)
