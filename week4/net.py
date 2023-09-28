import torch
import torch.nn as nn

class ConvBNAct(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, stride, padding):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_feat, out_feat, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_feat),
            nn.ReLU()
        )
    def forward(self, x):
        return self.body(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBNAct(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = ConvBNAct(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ConvBNAct(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.shortcut(x)  # Fixed the in-place operation here
        return out

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            ConvBNAct(3, 32, kernel_size=3, stride=1, padding=1),
            ResidualBlock(32, 32),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    # test LeNet
    net = LeNet()
    dummy_input = torch.randn(16, 3, 32, 32)
    dummy_output = net(dummy_input)
    print(f"# input shape = {dummy_input.shape}")
    print(f"# output shape = {dummy_output.shape}")
