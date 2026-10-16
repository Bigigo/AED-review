import torch
from torch import nn
from torch.nn import functional as F
from iaa_models import registry
from iaa_models.saan import SAAN

def stable_adain(content, style_stats):
    with torch.autocast(device_type=content.device.type, enabled=False):
        value = content.float()
        flat = value.flatten(2)
        mean = flat.mean(2)[:, :, None, None]
        std = (flat.std(2) + 1e-06)[:, :, None, None]
        style = style_stats.float()
        return style[:, :, 1, None, None] * (value - mean) / std + style[:, :, 0, None, None]

def stable_nonlocal(block, x):
    batch = x.shape[0]
    g = block.g(x).reshape(batch, block.inter_channels, -1).transpose(1, 2)
    theta = block.theta(x).reshape(batch, block.inter_channels, -1).transpose(1, 2)
    phi = block.phi(x).reshape(batch, block.inter_channels, -1)
    with torch.autocast(device_type=x.device.type, enabled=False):
        attention = F.softmax(theta.float() @ phi.float(), dim=-1)
        y = (attention @ g.float()).transpose(1, 2).contiguous()
        y = y.reshape(batch, block.inter_channels, *x.shape[2:])
    return block.W(y) + x

class SAANAED(SAAN):

    def __init__(self):
        super().__init__(1)
        self.predictor[-1] = nn.Identity()
        for parameter in self.GenAes.parameters():
            parameter.requires_grad_(False)

    def forward(self, x, style_stats=None):
        if style_stats is None:
            raise ValueError('Use the validated, frozen VGG19 original/HFlip style cache')
        branch = self.StyAes.model
        content = branch.layer2(branch.layer1(branch.maxpool(branch.relu(branch.bn1(branch.conv1(x))))))
        style = F.relu(stable_adain(content, style_stats))
        if self.StyAes.identity:
            raise ValueError('Official SAAN uses SAB(identity=False)')
        feature = torch.cat((self.max_pool(style), self.GenAes(x)), 1)
        feature = self.bn(self.avg_pool(stable_nonlocal(self.NLB, feature)))
        return self.predictor(feature.flatten(1)).reshape(-1)
