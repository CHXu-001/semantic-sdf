import torch
import torch.nn as nn
from torchvision import models


class SemanticContentModel(nn.Module):
    """
    This sementic ResNet model measures the layer-wise similarity of the generated
    and the paired data as the loss to supervise the training of translator model.
    """

    def __init__(self, content_layers: list[str]) -> None:
        """
        This method initializes the sementic content model and determines the set of
        layers specified by 'content_layers', of which the outputs are used to
        calculate the loss.
        """
        super().__init__()

        self._check_layers(content_layers)
        self.layers = content_layers
        resnet_model = models.resnet18(pretrained=True)

        self.conv1 = resnet_model.conv1
        self.bn1 = resnet_model.bn1
        self.relu = resnet_model.relu
        self.maxpool = resnet_model.maxpool
        self.layer1 = resnet_model.layer1
        self.layer2 = resnet_model.layer2
        self.layer3 = resnet_model.layer3
        self.layer4 = resnet_model.layer4

    def _check_layers(self, layers: list[str]) -> None:
        expected_layers = ["l0", "l1", "l2", "l3", "l4"]
        for layer in layers:
            if layer not in expected_layers:
                raise RuntimeError("Invalid layer number: {}".format(layer))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        out = {}
        out["l0"] = self.relu(self.bn1(self.conv1(x)))
        out["l1"] = self.layer1(self.maxpool(out["l0"]))
        out["l2"] = self.layer2(out["l1"])
        out["l3"] = self.layer3(out["l2"])
        out["l4"] = self.layer4(out["l3"])
        return [out[key] for key in self.layers]
