import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from torchvision.models import resnet50
from torch.nn.functional import elu, instance_norm

def get_pad(Input, ksize, stride, atrous=1):
    out = np.ceil(float(Input) / stride)
    return int(((out - 1) * stride + atrous * (ksize - 1) + 1 - Input) / 2)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class GatedConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,):
        super(GatedConv, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                           bias)
        self.sigmoid = torch.nn.Sigmoid()

    def gated(self, mask):
        # return torch.clamp(mask, -1, 1)
        return self.sigmoid(mask)

    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        x = elu(x) * self.gated(mask)
        x = instance_norm(x)
        return x 

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layer1 = GatedConv(3, 32, 5, 1, padding=get_pad(256, 5, 1))
        self.layer2 = GatedConv(32, 64, 3, 2, padding=get_pad(256, 4, 2))
        self.layer3 = GatedConv(64, 64, 3, 1, padding=get_pad(128, 3, 1))
        self.layer4 = GatedConv(64, 128, 3, 2, padding=get_pad(128, 4, 2))
        self.layer5 = GatedConv(128, 128, 3, 1, padding=get_pad(64, 3, 1))
        self.layer6 = GatedConv(128, 128, 3, 1, padding=get_pad(64, 3, 1))
        self.layer7 = GatedConv(128, 128, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2))
        self.layer8 = GatedConv(128, 128, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4))
        self.layer9 = GatedConv(128, 128, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8))
        self.layer10 = GatedConv(128, 128, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16))
        self.layer11 = GatedConv(128, 128, 3, 1, padding=get_pad(64, 3, 1))
        self.layer12 = GatedConv(256, 128, 3, 1, padding=get_pad(64, 3, 1))
        self.layer13 = nn.PixelShuffle(upscale_factor = 2) # replacing Upsampling
        self.layer14 = GatedConv(64, 64, 3, 1, padding=get_pad(128, 3, 1))
        self.layer15 = GatedConv(128, 64, 3, 1, padding=get_pad(128, 3, 1))
        self.layer16 = nn.PixelShuffle(upscale_factor = 2) # replacing Upsampling
        self.layer17 = GatedConv(32, 32, 3, 1, padding=get_pad(256, 3, 1))
        self.layer18 = GatedConv(64, 16, 3, 1, padding=get_pad(256, 3, 1))
        self.layer19 = nn.Conv2d(16, 1, kernel_size = 3, stride = 1, padding = 1)

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out = self.layer6(out5)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = torch.cat([out, out5], dim=1)
        out = self.layer12(out)
        out = torch.cat([out, out4], dim=1)
        out = self.layer13(out)
        out = self.layer14(out)
        out = torch.cat([out, out3], dim=1)
        out = self.layer15(out)
        out = torch.cat([out, out2], dim=1)
        out = self.layer16(out)
        out = self.layer17(out)
        out = torch.cat([out, out1], dim=1)
        out = self.layer18(out)
        out = self.layer19(out)
        return out

        

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(),
        )
        self.layer6 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(),
        )
        self.layer7 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(),
        )
        self.layer8 = Flatten()
        # self.layer9 = nn.Linear(4096, 256, bias=False)
        self.layer10 = nn.utils.spectral_norm(nn.Linear(1000, 256, bias=False))
        self.layer11 = nn.utils.spectral_norm(nn.Linear(256, 1, bias=False))

    def forward(self, input, mask, labels):
        out = torch.cat([input, mask], dim=1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        # out = self.layer9(out)
        out_t = self.layer11(out)
        z = self.layer10(labels)
        out = (out * z).sum(1, keepdim=True)
        out = torch.add(out, out_t)
        return out
        
        
class InceptionExtractor(nn.Module):
    def __init__(self):
        super(InceptionExtractor, self).__init__()
        self.resnet = resnet50(pretrained=True)
        self.resnet.conv1 = torch.nn.Conv1d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
        # self.inception_v3 = inception_v3(pretrained=True, transform_input=True, aux_logits=False)

    def forward(self, input):
        x = interpolate(input, (224, 224), mode='bilinear', align_corners=False)
        #x = self.inception_v3((x + 1) / 2) # normalize to [0,1]
        x = self.resnet((x + 1) / 2)
        x = torch.nn.functional.normalize(x) # normalize class output as proposed in paper
        return x
