import torch
import torch.nn as nn

def conv(in_planes, out_planes, kernel_size, stride=1, bias=True):
    return nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=(kernel_size//2), bias=bias)

def convT(in_planes, out_planes, kernel_size, stride=1, padding=1, bias=True):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride=stride, padding=padding, bias=bias)

class ResBlock(nn.Module):
    def __init__(self, in_planes=32, out_planes=32):
        super().__init__()
        self.conv1 = conv(in_planes, out_planes, 3)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv(in_planes, out_planes, 3)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return identity + x

class EDcoder(nn.Module):
    def __init__(self, level, num_resblocks, input_channels, c):
        super().__init__()
        # Conv
        # c = 8
        self.layer1 = conv(input_channels, c, kernel_size=3)
        modules = []
        for i in range(num_resblocks):
            modules.append(ResBlock(c*level, c*level))
        self.layer2 = nn.Sequential(*modules)
        # Conv
        self.layer3 = conv(c*level, c*2*level, kernel_size=3, stride=2)
        modules = []
        for i in range(num_resblocks):
            modules.append(ResBlock(c*2*level, c*2*level))
        self.layer4 = nn.Sequential(*modules)
        # Conv
        self.layer5 = conv(c*2*level, c*4*level, kernel_size=3, stride=2)
        modules = []
        self.layerff = conv(c*4*(2*level-1), c*4*level, kernel_size=1)
        for i in range(num_resblocks):
            modules.append(ResBlock(c*4*level, c*4*level))
        self.layer6 = nn.Sequential(*modules)
        self.layer7 = convT(c*4*level, c*2*level, kernel_size=4, stride=2)
        modules = []
        for i in range(num_resblocks):
            modules.append(ResBlock(c*2*level, c*2*level))
        self.layer8 = nn.Sequential(*modules)
        # Deconv
        self.layer9 = convT(c*2*level, c*level, kernel_size=4, stride=2)
        modules = []
        for i in range(num_resblocks):
            modules.append(ResBlock(c*level, c*level))
        self.layer10 = nn.Sequential(*modules)
        # Conv
        self.layer11 = conv(c*level, c, kernel_size=3)

    def forward(self, x, d=None, f=None):
        # Conv
        x = self.layer1(x)
        if d != None:
            x = torch.cat((x, d), 1)
        x2 = self.layer2(x)
        # Conv
        x = self.layer3(x2)
        x4 = self.layer4(x)
        # Conv
        x = self.layer5(x4)
        if f != None:
            x = torch.cat((x, f), 1)
        x = self.layerff(x)
        x6 = self.layer6(x)
        # Deconv
        x = self.layer7(x6)
        x = x + x4
        x = self.layer8(x)
        # Deconv
        x = self.layer9(x)
        x = x + x2
        x = self.layer10(x)
        # Conv
        x = self.layer11(x)
        return x, x6

class model(nn.Module):
    def __init__(self, num_resblocks, input_channels):
        super().__init__()
        self.edcoder1 = EDcoder(level=1, num_resblocks=num_resblocks[0], input_channels=input_channels[0], c=8)
        self.edcoder2 = EDcoder(level=2, num_resblocks=num_resblocks[1], input_channels=input_channels[1], c=8)
        self.edcoder3 = EDcoder(level=3, num_resblocks=num_resblocks[2], input_channels=input_channels[2], c=8)
        self.edcoder4 = EDcoder(level=4, num_resblocks=num_resblocks[3], input_channels=input_channels[3], c=8)
        self.conv_output = conv(8, 3, kernel_size=3)

    def forward(self, x):
        d1, f1 = self.edcoder1(x)
        d2, f2 = self.edcoder2(x, d1, f1)
        d3, f3 = self.edcoder3(x, torch.cat((d1, d2), 1), f2)
        d4, f4 = self.edcoder4(x, torch.cat((d1, d2, d3), 1), f3)
        output = self.conv_output(d4)

        return output