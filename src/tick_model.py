"""Small 1D-CNN over a fixed-length tick window (price, size, side, dt as
4 input channels), predicting next-tick direction. Deep enough that a
training step does real, non-trivial GPU work -- the point of this
benchmark is the input path feeding it, not the model itself."""
import torch.nn as nn


class ConvBlock1d(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class TickCNN(nn.Module):
    def __init__(self, in_channels=4, width=64, num_classes=2):
        super().__init__()
        self.stem = ConvBlock1d(in_channels, width)
        self.stage1 = nn.Sequential(ConvBlock1d(width, width), ConvBlock1d(width, width))
        self.stage2 = nn.Sequential(
            ConvBlock1d(width, width * 2, stride=2), ConvBlock1d(width * 2, width * 2)
        )
        self.stage3 = nn.Sequential(
            ConvBlock1d(width * 2, width * 4, stride=2), ConvBlock1d(width * 4, width * 4)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(width * 4, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)
