import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights, resnet50, ResNet50_Weights, resnet18
import torch.nn.functional as F
from .saan_layers import adain, NonLocalBlock, Pool2x2

class VGG(nn.Module):

    def __init__(self):
        super(VGG, self).__init__()
        from torchvision.models.vgg import make_layers, cfgs
        self.model = nn.Sequential(*make_layers(cfgs['E'])[:21])
        self._freeze_params()

    def forward(self, x):
        x = self.model(x)
        return x

    def _freeze_params(self):
        for p in self.model.parameters():
            p.requires_grad = False

class SAB(nn.Module):

    def __init__(self, identity=False):
        super(SAB, self).__init__()
        model = resnet50()
        self.model = nn.Module()
        self.model.conv1 = model.conv1
        self.model.bn1 = model.bn1
        self.model.relu = model.relu
        self.model.maxpool = model.maxpool
        self.model.layer1 = model.layer1
        self.model.layer2 = model.layer2
        self.identity = identity
        self.vgg = VGG()
        self._init_weights()

    def forward(self, x, style_stats=None):
        sty = self.vgg(x) if style_stats is None else None
        aes = self.model.conv1(x)
        aes = self.model.bn1(aes)
        aes = self.model.relu(aes)
        aes = self.model.maxpool(aes)
        aes = self.model.layer1(aes)
        aes = self.model.layer2(aes)
        output = adain(aes, sty, style_stats)
        if self.identity:
            output += aes
        return F.relu(output)

    def _init_weights(self):
        pass

class GAB(nn.Module):

    def __init__(self):
        super(GAB, self).__init__()
        model = resnet50()
        self.model = nn.Module()
        self.model.conv1 = model.conv1
        self.model.bn1 = model.bn1
        self.model.relu = model.relu
        self.model.maxpool = model.maxpool
        self.model.layer1 = model.layer1
        self.model.layer2 = model.layer2
        self.model.layer3 = model.layer3
        self._init_weights()

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        return x

    def _init_weights(self):
        pass

class SAAN(nn.Module):

    def __init__(self, num_classes) -> None:
        super().__init__()
        self.GenAes = GAB()
        self.StyAes = SAB()
        self.NLB = NonLocalBlock(in_channels=1536)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = Pool2x2()
        self.bn = nn.BatchNorm2d(num_features=1536)
        self.predictor = nn.Sequential(nn.Linear(1536 * 4, 2048), nn.ReLU(inplace=True), nn.Dropout(p=0.5), nn.Linear(2048, num_classes), nn.Sigmoid())
        self._initial_weights()

    def forward(self, x, style_stats=None):
        gen_aes = self.GenAes(x)
        sty_aes = self.StyAes(x, style_stats)
        sty_aes = self.max_pool(sty_aes)
        all_aes = torch.cat((sty_aes, gen_aes), 1)
        all_aes = self.NLB(all_aes)
        all_aes = self.avg_pool(all_aes)
        all_aes = self.bn(all_aes)
        fc_input = torch.flatten(all_aes, start_dim=1)
        output = self.predictor(fc_input)
        return output

    def _initial_weights(self):
        for m in self.bn.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
                nn.init.constant_(m.bias.data, 0.0)
        for m in self.predictor.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
                nn.init.constant_(m.bias.data, 0.0)
