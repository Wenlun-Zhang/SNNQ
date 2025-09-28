import torch.nn as nn
from model.ResNet import BasicBlock
from model.VGG import VGG
from snnq.quantization.quantized_module import QuantizedLayer, QuantizedBlock, Quantizer   # noqa: F401


class QuantBasicBlock(QuantizedBlock):
    def __init__(self, org_module: BasicBlock, w_qconfig):
        super().__init__()

        res_conv1 = QuantizedLayer(org_module.residual_function[0], None, w_qconfig)
        res_bn1 = org_module.residual_function[1]
        res_relu = org_module.residual_function[2]
        res_conv2 = QuantizedLayer(org_module.residual_function[3], None, w_qconfig)
        res_bn2 = org_module.residual_function[4]
        self.residual_function = nn.Sequential(
            res_conv1,
            res_bn1,
            res_relu,
            res_conv2,
            res_bn2
        )

        self.shortcut = org_module.shortcut
        if len(list(org_module.shortcut)) != 0:
            sc_conv1 = QuantizedLayer(org_module.shortcut[0], None, w_qconfig)
            sc_bn1 = org_module.shortcut[1]
            self.shortcut = nn.Sequential(
                sc_conv1,
                sc_bn1
            )

        self.relu = org_module.relu

    def forward(self, x):
        return self.relu(self.residual_function(x) + self.shortcut(x))


class QuantVGGBlock(QuantizedBlock):
    def __init__(self, org_block: nn.Sequential, w_qconfig):
        super().__init__()
        quant_layers = []
        for m in org_block:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                quant_layers.append(QuantizedLayer(m, None, w_qconfig))
            else:
                quant_layers.append(m)
        self.block = nn.Sequential(*quant_layers)

    def forward(self, x):
        return self.block(x)


class QuantVGG(nn.Module):
    def __init__(self, org_vgg: VGG, w_qconfig):
        super().__init__()
        self.layer1 = QuantVGGBlock(org_vgg.layer1, w_qconfig)
        self.layer2 = QuantVGGBlock(org_vgg.layer2, w_qconfig)
        self.layer3 = QuantVGGBlock(org_vgg.layer3, w_qconfig)
        self.layer4 = QuantVGGBlock(org_vgg.layer4, w_qconfig)
        self.layer5 = QuantVGGBlock(org_vgg.layer5, w_qconfig)
        self.classifier = QuantVGGBlock(org_vgg.classifier, w_qconfig)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.classifier(x)
        return x


specials = {
    BasicBlock: QuantBasicBlock,
    VGG: QuantVGG
}
