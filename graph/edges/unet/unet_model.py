""" Full assembly of the parts to form the complete network """
import torch
import torch.nn.functional as F

from .unet_parts import *


class DropBlock2D(nn.Module):
    r"""Randomly zeroes spatial blocks of the input tensor.
    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        keep_prob (float, optional): probability of an element to be kept.
        Authors recommend to linearly decrease this value from 1 to desired
        value.
        block_size (int, optional): size of the block. Block size in paper
        usually equals last feature map dimensions.
    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """
    def __init__(self, keep_prob=0.9, block_size=7):
        super(DropBlock2D, self).__init__()
        self.keep_prob = keep_prob
        self.block_size = block_size

    def forward(self, input):
        if not self.training or self.keep_prob == 1:
            return input
        gamma = (1. - self.keep_prob) / self.block_size**2
        for sh in input.shape[2:]:
            gamma *= sh / (sh - self.block_size + 1)
        M = torch.bernoulli(torch.ones_like(input) * gamma)
        Msum = F.conv2d(M,
                        torch.ones((input.shape[1], 1, self.block_size,
                                    self.block_size)).to(device=input.device,
                                                         dtype=input.dtype),
                        padding=self.block_size // 2,
                        groups=input.shape[1])
        torch.set_printoptions(threshold=5000)
        mask = (Msum < 1).to(device=input.device, dtype=input.dtype)
        return input * mask * mask.numel() / mask.sum()


def get_unet(model_type,
             n_channels,
             n_classes,
             from_exp,
             to_exp,
             dropout_prob=0,
             dropout_encoder=False,
             dropout_decoder=False,
             dropblock_prob=0,
             dropblock_encoder=False,
             dropblock_decoder=False):
    assert (model_type in [0, 1, 2])
    if model_type == 0:
        return UNetGood(n_channels=n_channels,
                        n_classes=n_classes,
                        from_exp=from_exp,
                        to_exp=to_exp)
    elif model_type == 1:
        return UNetMedium(n_channels=n_channels,
                          n_classes=n_classes,
                          from_exp=from_exp,
                          to_exp=to_exp,
                          with_dropout=False,
                          dropout_prob=dropout_prob,
                          dropout_encoder=dropout_encoder,
                          dropout_decoder=dropout_decoder,
                          dropblock_prob=dropblock_prob,
                          dropblock_encoder=dropblock_encoder,
                          dropblock_decoder=dropblock_decoder)
    elif model_type == 2:
        return UNetMedium(n_channels=n_channels,
                          n_classes=n_classes,
                          from_exp=from_exp,
                          with_dropout=True,
                          to_exp=to_exp)


class UNetMedium(nn.Module):
    def __init__(self,
                 n_channels,
                 n_classes,
                 from_exp,
                 to_exp,
                 with_dropout=False,
                 dropout_prob=0,
                 dropout_encoder=False,
                 dropout_decoder=False,
                 dropblock_prob=0,
                 dropblock_encoder=False,
                 dropblock_decoder=False):
        # 4 mil params
        super(UNetMedium, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        bilinear = True
        self.to_exp = to_exp

        self.inc = DoubleConv(n_channels, 32, with_dropout=with_dropout)
        self.down1 = Down(32, 64, with_dropout=with_dropout)
        self.down2 = Down(64, 128, with_dropout=with_dropout)
        self.down3 = Down(128, 256, with_dropout=with_dropout)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear, with_dropout=with_dropout)
        self.up2 = Up(256, 128 // factor, bilinear, with_dropout=with_dropout)
        self.up3 = Up(128, 64 // factor, bilinear, with_dropout=with_dropout)
        self.up4 = Up(64, 32, bilinear, with_dropout=with_dropout)
        self.outc = OutConv(32, n_classes)

        if dropout_prob > 0:
            if dropout_encoder:
                self.drop_layer_encoder = torch.nn.Dropout(dropout_prob)
            else:
                self.drop_layer_encoder = lambda x: x
            if dropout_decoder:
                self.drop_layer_decoder = torch.nn.Dropout(dropout_prob)
            else:
                self.drop_layer_decoder = lambda x: x

        elif dropblock_prob > 0:
            dropblock_prob = 1 - dropblock_prob
            if dropblock_encoder:
                self.drop_layer_encoder = DropBlock2D(dropblock_prob)
            else:
                self.drop_layer_encoder = lambda x: x
            if dropblock_decoder:
                self.drop_layer_decoder = DropBlock2D(dropblock_prob)
            else:
                self.drop_layer_decoder = lambda x: x

        else:
            self.drop_layer_encoder = lambda x: x
            self.drop_layer_decoder = lambda x: x

        #self.dropblock = DropBlock2D(keep_prob=0.75)
        #self.dropout = torch.nn.Dropout(0.25)

    def forward(self, inp):
        x, postproc_fcn = inp
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.drop_layer_encoder(x2)

        x3 = self.down2(x2)
        x3 = self.drop_layer_encoder(x3)

        x4 = self.down3(x3)
        x4 = self.drop_layer_encoder(x4)

        x5 = self.down4(x4)
        x5 = self.drop_layer_encoder(x5)

        x = self.up1(x5, x4)
        x = self.drop_layer_decoder(x)
        x = self.up2(x, x3)
        x = self.drop_layer_decoder(x)
        x = self.up3(x, x2)
        x = self.drop_layer_decoder(x)
        x = self.up4(x, x1)
        x = self.drop_layer_decoder(x)
        logits = self.outc(x)
        return postproc_fcn(logits)


class UNetGood(nn.Module):
    def __init__(self, n_channels, n_classes, from_exp, to_exp):
        # 1 mil params
        super(UNetGood, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        bilinear = True
        self.to_exp = to_exp

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down4 = Down(128, 256 // factor)
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32 // factor, bilinear)
        self.up4 = Up(32, 16, bilinear)
        self.outc = OutConv(16, n_classes)

    def forward(self, inp):
        x, postproc_fcn = inp
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return postproc_fcn(logits)


# class UNetLarge(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=True):
#         # 17 mil params
#         super(UNetLarge, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         self.inc = DoubleConv(n_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         factor = 2 if bilinear else 1
#         self.down4 = Down(512, 1024 // factor)
#         self.up1 = Up(1024, 512 // factor, bilinear)
#         self.up2 = Up(512, 256 // factor, bilinear)
#         self.up3 = Up(256, 128 // factor, bilinear)
#         self.up4 = Up(128, 64, bilinear)
#         self.outc = OutConv(64, n_classes)

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         logits = self.outc(x)
#         return logits

# class UNetSmall(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=True):
#         # 200k params
#         super(UNetSmall, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         self.inc = DoubleConv(n_channels, 32)
#         self.down1 = Down(32, 64)
#         factor = 2 if bilinear else 1
#         self.down2 = Down(64, 128 // factor)
#         self.up1 = Up(128, 64 // factor, bilinear)
#         self.up2 = Up(64, 32, bilinear)
#         self.outc = OutConv(32, n_classes)

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x = self.up1(x3, x2)
#         x = self.up2(x, x1)
#         logits = self.outc(x)
#         return logits
