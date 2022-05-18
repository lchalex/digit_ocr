import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck

class DecoderBlock(nn.Module):
    def __init__(
            self,
            skip_channels,
            lower_channels
    ):
        super().__init__()

        self.block1 = Bottleneck(
            inplanes=lower_channels,
            planes=skip_channels // 4,
            downsample=nn.Conv2d(lower_channels, skip_channels, kernel_size=1, bias=False)
        )

        self.block2 = Bottleneck(
            inplanes=skip_channels,
            planes=skip_channels // 4
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=(skip.size()[2], skip.size()[3]), mode="bilinear", align_corners=True)

        x = self.block1(x)
        s = x + skip
        s = self.block2(s)
        return s

class CenterBlock(nn.Sequential):
    def __init__(self, in_channels):

        block = Bottleneck(
            inplanes=in_channels,
            planes=in_channels // 4,
            downsample=nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        )

        super().__init__(block)

class FPNDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels
    ):
        super().__init__()

        self.encoder_channels = encoder_channels

        pyramid_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = pyramid_channels[0]
        skip_channels = pyramid_channels[1:]
        lower_channels = pyramid_channels[:-1]

        self.center = CenterBlock(head_channels)

        # combine decoder keyword arguments
        blocks = [
            DecoderBlock(skip_ch, low_ch)
            for skip_ch, low_ch in zip(skip_channels, lower_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def get_out_channels(self):
        return self.encoder_channels[::-1]

    def forward(self, *features):

        features = features[::-1]  # reverse channels to start from head of encoder

        out = []
        skips = features[1:]

        x = self.center(features[0])
        out.append(x)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i]
            x = decoder_block(x, skip)
            out.append(x)

        return out