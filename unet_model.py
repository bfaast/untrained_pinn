# full assembly of the sub-parts to form the complete net
from unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        # remember if you are using bilinear interpolation or convtranspose2d for upsampling
        # when using bilinear interpolation, in channels for up1 is 256+128
        # when using convtranspose2d, in channels for up1 is 256, up2 is 128
        self.up1 = up(384, 128)
        self.up2 = up(192, 64)
        self.up3 = up(96, 32)

        self.outc4 = outconvpadded(32, out_ch=32)
        self.outc5 = outconv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc4(x)
        x = self.outc5(x)
        return x
