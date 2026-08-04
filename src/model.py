"""Small CNN classifier -- deep enough to be GPU-bound at reasonable batch
sizes, shallow enough to train fast on a single consumer GPU for this
project's purposes (demonstrating DDP/AMP/grad-accum mechanics, not
chasing state-of-the-art accuracy)."""
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SmallCNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, width=64):
        super().__init__()
        self.stem = ConvBlock(in_channels, width)
        self.stage1 = nn.Sequential(ConvBlock(width, width), ConvBlock(width, width))
        self.stage2 = nn.Sequential(ConvBlock(width, width * 2, stride=2), ConvBlock(width * 2, width * 2))
        self.stage3 = nn.Sequential(ConvBlock(width * 2, width * 4, stride=2), ConvBlock(width * 4, width * 4))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width * 4, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)
