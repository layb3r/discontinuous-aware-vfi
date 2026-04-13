import torch
import torch.nn as nn
from helper import ResidualBlock, DownSampleBlock, UpSampleBlock, GroupNorm, Swish

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.Elayer1 = nn.Sequential(ResidualBlock(128, 128),
                                     ResidualBlock(128, 128),
                                     DownSampleBlock(128))#/4
        self.Elayer2 = nn.Sequential(ResidualBlock(128, 256),
                                     ResidualBlock(256, 256),
                                     DownSampleBlock(256))#/8
        self.Elayer3 = nn.Sequential(ResidualBlock(256, 256),
                                     ResidualBlock(256, 256),
                                     DownSampleBlock(256))#/16
        self.Elayer4 = nn.Sequential(ResidualBlock(256, 512),
                                     ResidualBlock(512, 512))
        self.Efinal =  nn.Sequential(ResidualBlock(512, 512),
                                     ResidualBlock(512, 512),
                                     GroupNorm(512),
                                     Swish(),
                                     nn.Conv2d(512, 256, 3, 1, 1))
        
        self.Dhead = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1),
                                   ResidualBlock(512, 512),
                                   ResidualBlock(512, 512))
        
        self.Dlayer0 = nn.Sequential(ResidualBlock(512, 512),
                                     ResidualBlock(512, 512))#/16
        
        self.Dlayer1 = nn.Sequential(ResidualBlock(512+256, 256),
                                     ResidualBlock(256, 256),
                                     UpSampleBlock(256))#/8
        
        self.Dlayer2 = nn.Sequential(ResidualBlock(256+256, 256),
                                     ResidualBlock(256, 256),
                                     UpSampleBlock(256))#/4
        
        self.Dlayer3 = nn.Sequential(ResidualBlock(256+128, 128),
                                     ResidualBlock(128, 128),
                                     UpSampleBlock(128))#/2
        
        self.Dlayer4 = nn.Sequential(ResidualBlock(128+128, 128),
                                     ResidualBlock(128, 128),
                                     UpSampleBlock(128))#/1
        
        self.Dfinal = nn.Sequential(GroupNorm(128),
                                    Swish(),
                                    nn.Conv2d(128, 3, 3, 1, 1))
        
    def forward(self, x):
        out0 = self.Elayer1(x)#/4
        out1 = self.Elayer2(out0)#/8
        out2 = self.Elayer3(out1)#/16
        out = self.Elayer4(out2)#/16
        out = self.Efinal(out)#/16
        out = self.Dhead(out)#/16
        out = self.Dlayer0(out)#/16
        out = self.Dlayer1(torch.cat([out, out2], dim=1))#/8
        out = self.Dlayer2(torch.cat([out, out1], dim=1))#/4
        out = self.Dlayer3(torch.cat([out, out0], dim=1))#/2
        out = self.Dlayer4(torch.cat([out,    x], dim=1))#/1
        out = self.Dfinal(out)
        return out

class LiteUnet(nn.Module):
    def __init__(self):
        super(LiteUnet, self).__init__()
        self.Elayer1 = nn.Sequential(ResidualBlock(128, 32),
                                     DownSampleBlock(32))#/2
        self.Elayer2 = nn.Sequential(ResidualBlock(32, 64),
                                     DownSampleBlock(64))#/4
        self.Elayer3 = nn.Sequential(ResidualBlock(64, 128),
                                     DownSampleBlock(128))#/8
        self.Elayer4 = nn.Sequential(ResidualBlock(128, 128))#/8
        
        self.Dlayer0 = nn.Sequential(ResidualBlock(128, 64))#/8
        
        self.Dlayer1 = nn.Sequential(ResidualBlock(64+128, 64),
                                     UpSampleBlock(64))#/4
        
        self.Dlayer2 = nn.Sequential(ResidualBlock(64+64, 128),
                                     UpSampleBlock(128))#/2
        
    def forward(self, x):
        out0 = self.Elayer1(x)#/2
        out1 = self.Elayer2(out0)#/4
        out2 = self.Elayer3(out1)#/8
        out = self.Elayer4(out2)#/8
        out = self.Dlayer0(out)#/8
        out = self.Dlayer1(torch.cat([out, out2], dim=1))
        out = self.Dlayer2(torch.cat([out, out1], dim=1))
        return out