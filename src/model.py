import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from config import cfg
import albumentations as album
import segmentation_models_pytorch as smp
from torchsummary import summary


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1).cuda(),
            nn.BatchNorm2d(out_channels).cuda(),
            nn.ReLU(inplace=True).cuda(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1).cuda(),
            nn.BatchNorm2d(out_channels).cuda(),
            nn.ReLU(inplace=True).cuda()
        )

    def forward(self, x):
        return self.conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.inc = DoubleConv(n_channels, 4*2).cuda()
        self.down1 = DoubleConv(4*2, 8*2).cuda()
        self.down2 = DoubleConv(8*2, 16*2).cuda()
        self.down3 = DoubleConv(16*2, 32*2).cuda()
        self.down4 = DoubleConv(32*2, 64*2).cuda()
        self.up1 = Up(64*2, 32*2).cuda()
        self.up2 = Up(32*2, 16*2).cuda()
        self.up3 = Up(16*2, 8*2).cuda()
        self.up4 = Up(8*2, 4*2).cuda()
        self.outc = nn.Conv2d(4*2, n_classes, kernel_size=1).cuda()

        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(F.max_pool2d(x1, kernel_size=2, stride=2))
        x3 = self.down2(F.max_pool2d(x2, kernel_size=2, stride=2))
        x4 = self.down3(F.max_pool2d(x3, kernel_size=2, stride=2))
        x5 = self.down4(F.max_pool2d(x4, kernel_size=2, stride=2))
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        logits = self.softmax(logits)
        return logits
    
def get_unet():
    model = UNet(n_channels=3, n_classes=2)
    summary(model, (3, 1024, 1024))

    return model

def get_unet_pretrained():
    ENCODER = 'resnet50'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation

    # create segmentation model with pretrained encoder
    model = smp.Unet(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=cfg.DATASET.NBR_CLASSE, 
        activation=ACTIVATION,
    ).cuda()

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    return model

def get_unet_complete_pretrained():
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True).cuda()
    model.conv = torch.nn.Conv2d(32, 2, kernel_size=(1, 1), stride=(1, 1)).cuda()

    summary(model, (3, 1024, 1024))
    return model


class DiscriminatorConv(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(2, 8, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(8, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(16, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        ).cuda()
        self.fc = nn.Sequential(
            nn.Linear(32768, 128),
            nn.Linear(128, 2),
            nn.Sigmoid()
        ).cuda()

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)
        return out

def get_discriminator():
    model = DiscriminatorConv().cuda()

    summary(model, (2, 1024, 1024))
    return model
