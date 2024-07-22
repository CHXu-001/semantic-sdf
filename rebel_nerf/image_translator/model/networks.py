import math

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18


class UpsampleBasicBlock(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        scale: int = 2,
        mode: str = "bilinear",
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.InstanceNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels=planes,
            out_channels=planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.InstanceNorm2d(planes)

        kernel_size, padding = 3, 1
        if inplanes != planes:
            kernel_size, padding = 1, 0

        self.upsample = nn.Sequential(
            nn.Conv2d(
                inplanes,
                planes,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=False,
            ),
            nn.InstanceNorm2d(planes),
        )

        self.scale = scale
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(
            x, scale_factor=self.scale, mode=self.mode, align_corners=True
        )
        residual = self.upsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class TranslatorUpsampleResidual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        resnet = resnet18(pretrained=True)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self._build_upsample_layers()

    def _init_conv(self, module: nn.Module) -> None:
        out_channels, _, kernel_size0, kernel_size1 = module.weight.size()
        n = kernel_size0 * kernel_size1 * out_channels
        module.weight.data.normal_(0, math.sqrt(2.0 / n))

    def _init_bn(self, module: nn.Module) -> None:
        module.weight.data.fill_(1)
        module.bias.data.zero_()

    def _init_weights(self, block: nn.Module) -> None:
        for _, module in block.named_modules():
            if isinstance(module, nn.Conv2d):
                self._init_conv(module)
            elif isinstance(module, nn.BatchNorm2d):
                self._init_bn(module)

    def _conv_norm_relu(
        self,
        dim_in: int,
        dim_out: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_bias: bool = False,
    ) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(
                dim_in,
                dim_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=use_bias,
            ),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
        )

    def _build_upsample_layers(self) -> None:
        self.up1 = UpsampleBasicBlock(
            inplanes=512, planes=256, kernel_size=1, padding=0
        )
        self.up2 = UpsampleBasicBlock(
            inplanes=256, planes=128, kernel_size=1, padding=0
        )
        self.up3 = UpsampleBasicBlock(inplanes=128, planes=64, kernel_size=1, padding=0)
        self.up4 = UpsampleBasicBlock(inplanes=64, planes=64, kernel_size=3, padding=1)

        self.skip_3 = self._conv_norm_relu(
            dim_in=256, dim_out=256, kernel_size=1, padding=0
        )
        self.skip_2 = self._conv_norm_relu(
            dim_in=128, dim_out=128, kernel_size=1, padding=0
        )
        self.skip_1 = self._conv_norm_relu(
            dim_in=64, dim_out=64, kernel_size=1, padding=0
        )
        self.up_image = nn.Sequential(nn.Conv2d(64, 3, 7, 1, 3, bias=False), nn.Tanh())

        # init weights.
        self._init_weights(self.up1)
        self._init_weights(self.up2)
        self._init_weights(self.up3)
        self._init_weights(self.up4)
        self._init_weights(self.skip_3)
        self._init_weights(self.skip_2)
        self._init_weights(self.skip_1)
        self._init_weights(self.up_image)

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        out = {}

        out["0"] = self.relu(self.bn1(self.conv1(input)))
        out["1"] = self.layer1(out["0"])
        out["2"] = self.layer2(out["1"])
        out["3"] = self.layer3(out["2"])
        out["4"] = self.layer4(out["3"])

        skip1 = self.skip_1(out["1"])
        skip2 = self.skip_2(out["2"])
        skip3 = self.skip_3(out["3"])

        upconv4 = self.up1(out["4"])
        upconv3 = self.up2(upconv4 + skip3)
        upconv2 = self.up3(upconv3 + skip2)
        upconv1 = self.up4(upconv2 + skip1)

        gen_img = self.up_image(upconv1)

        return gen_img
