import torch

from torch import nn
from torch.nn import init


class DemoClassifier(nn.Module):
    def __init__(self, channels):
        super(DemoClassifier, self).__init__()

        conv_layers = list()

        # first conv block
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=(
            5, 5), stride=(2, 2), padding=(2, 2))
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu, self.bn1]

        # second conv block
        self.conv2 = nn.Conv2d(64, 32, kernel_size=(
            3, 3), stride=(2, 2), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu, self.bn2]

        # third conv block
        self.conv3 = nn.Conv2d(32, 16, kernel_size=(
            3, 3), stride=(2, 2), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu, self.bn3]

        # fourth conv block
        self.conv4 = nn.Conv2d(16, 8, kernel_size=(
            3, 3), stride=(2, 2), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu, self.bn4]

        # lineaer classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(in_features=8, out_features=2)

        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.conv(x)
        x = self.ap(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x
