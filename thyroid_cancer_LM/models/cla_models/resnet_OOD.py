import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channel, f, filters, s):
        super(ConvBlock, self).__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.Conv2d(in_channel, F1, 1, stride=s, padding=0, bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(True),
            nn.Conv2d(F1, F2, f, stride=1, padding=True, bias=False),
            nn.BatchNorm2d(F2),
            nn.ReLU(True),
            nn.Conv2d(F2, F3, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F3),
        )
        self.shortcut_1 = nn.Conv2d(in_channel, F3, 1, stride=s, padding=0, bias=False)
        self.batch_1 = nn.BatchNorm2d(F3)
        self.relu_1 = nn.ReLU(True)

    def forward(self, X):
        X_shortcut = self.shortcut_1(X)
        X_shortcut = self.batch_1(X_shortcut)
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X


class IndentityBlock(nn.Module):
    def __init__(self, in_channel, f, filters):
        super(IndentityBlock, self).__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.Conv2d(in_channel, F1, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(True),
            nn.Conv2d(F1, F2, f, stride=1, padding=True, bias=False),
            nn.BatchNorm2d(F2),
            nn.ReLU(True),
            nn.Conv2d(F2, F3, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F3),
        )
        self.relu_1 = nn.ReLU(True)

    def forward(self, X):
        X_shortcut = X
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X


class ResModel(nn.Module):
    def __init__(self, n_class):
        super(ResModel, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, padding=1),
        )
        self.stage2 = nn.Sequential(
            ConvBlock(64, f=3, filters=[64, 64, 256], s=1),
            IndentityBlock(256, 3, [64, 64, 256]),
            IndentityBlock(256, 3, [64, 64, 256]),
        )
        self.stage3 = nn.Sequential(
            ConvBlock(256, f=3, filters=[128, 128, 512], s=2),
            IndentityBlock(512, 3, [128, 128, 512]),
            IndentityBlock(512, 3, [128, 128, 512]),
            IndentityBlock(512, 3, [128, 128, 512]),
        )
        self.stage4 = nn.Sequential(
            ConvBlock(512, f=3, filters=[256, 256, 1024], s=2),
            IndentityBlock(1024, 3, [256, 256, 1024]),
            IndentityBlock(1024, 3, [256, 256, 1024]),
            IndentityBlock(1024, 3, [256, 256, 1024]),
            IndentityBlock(1024, 3, [256, 256, 1024]),
            IndentityBlock(1024, 3, [256, 256, 1024]),
        )
        self.stage5 = nn.Sequential(
            ConvBlock(1024, f=3, filters=[512, 512, 2048], s=2),
            IndentityBlock(2048, 3, [512, 512, 2048]),
            IndentityBlock(2048, 3, [512, 512, 2048]),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(2048, n_class)
        )
        self.conv_channel = nn.Conv2d(64, 1, kernel_size=1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        self.conv_cat_12 = nn.Conv2d(64 + 256, 256, 1)
        self.conv_cat_23 = nn.Conv2d(256 + 512, 512, 1)
        self.conv_cat_34 = nn.Conv2d(512 + 1024, 1024, 1)

    def forward(self, X):
        stage1_out = self.stage1(X)
        out_gram = stage1_out
        out_gram = self.conv_channel(out_gram)

        stage2_out = self.stage2(stage1_out)
        stage2_out = torch.cat([stage2_out, stage1_out], dim=1)
        stage2_out = self.conv_cat_12(stage2_out)

        stage3_out = self.stage3(stage2_out)
        stage3_out = torch.cat([stage3_out, self.maxpool(stage2_out)], dim=1)
        stage3_out = self.conv_cat_23(stage3_out)

        stage4_out = self.stage4(stage3_out)
        stage4_out = torch.cat([stage4_out, self.maxpool(stage3_out)], dim=1)
        stage4_out = self.conv_cat_34(stage4_out)

        stage5_out = self.stage5(stage4_out)
        out = self.pool(stage5_out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out, [out_gram]


if __name__ == '__main__':
    input = torch.ones((16, 3, 224, 224))
    model = ResModel(n_class=2)
    result = model(input)
    print(result)
