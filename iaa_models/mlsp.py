import torch
from torch import nn

def area_weights(source, target, device):
    edges = torch.arange(target + 1, device=device, dtype=torch.float32) * (source / target)
    cells = torch.arange(source, device=device, dtype=torch.float32)
    overlap = (torch.minimum(edges[1:, None], cells[None] + 1) - torch.maximum(edges[:-1, None], cells[None])).clamp(min=0)
    return overlap / (source / target)

def area_resize(x, size=5):
    h = area_weights(x.shape[-2], size, x.device)
    w = area_weights(x.shape[-1], size, x.device)
    return torch.matmul(torch.matmul(h, x.float()), w.T)

class MLSPExtractor(nn.Module):

    def __init__(self):
        super().__init__()
        from .inception_resnet_v2_offline import InceptionResnetV2
        self.backbone = InceptionResnetV2()
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self._pools = []
        self._hooks = []
        for name in ('mixed_5b', 'mixed_6a', 'mixed_7a'):
            self._hooks.append(getattr(self.backbone, name).register_forward_hook(self._mixed))
        for block in [*self.backbone.repeat, *self.backbone.repeat_1, *self.backbone.repeat_2, self.backbone.block8]:
            self._hooks.append(block.conv2d.register_forward_pre_hook(self._pre_projection))

    def _mixed(self, module, args, output):
        self._pools.append(area_resize(output))

    def _pre_projection(self, module, args):
        self._pools.append(area_resize(args[0]))

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, x):
        self._pools = []
        self.backbone.forward_features(x)
        if len(self._pools) != 43:
            raise RuntimeError('InceptionResNetV2 topology changed: expected 43 mixed tensors')
        features = torch.cat(self._pools, dim=1)
        self._pools = []
        if features.shape[1:] != (16928, 5, 5):
            raise RuntimeError(f'MLSP shape mismatch: {features.shape}')
        return features

class ConvBN(nn.Sequential):

    def __init__(self, channels, filters, kernel=1):
        shape = (kernel, kernel) if isinstance(kernel, int) else kernel
        conv = nn.Conv2d(channels, filters, shape, padding=(shape[0] // 2, shape[1] // 2), bias=False)
        nn.init.xavier_uniform_(conv.weight)
        bn = nn.BatchNorm2d(filters, eps=0.001, momentum=0.01)
        bn.weight.requires_grad_(False)
        super().__init__(conv, bn, nn.ReLU())

class MLSPHead(nn.Module):

    def __init__(self):
        super().__init__()
        self.branch1 = ConvBN(16928, 1024)
        self.branch3 = ConvBN(16928, 1024)
        self.branch3a = ConvBN(1024, 512, (1, 3))
        self.branch3b = ConvBN(1024, 512, (3, 1))
        self.branch_pool = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False), ConvBN(16928, 1024))
        layers = []
        dimensions, dropout = ([3072, 2048, 1024, 256, 1], [0.25, 0.25, 0.5, 0])
        for i, (before, after) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            linear = nn.Linear(before, after)
            nn.init.kaiming_normal_(linear.weight, nonlinearity='relu')
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            if i < 3:
                layers.extend([nn.ReLU(), nn.BatchNorm1d(after, eps=0.001, momentum=0.01), nn.Dropout(dropout[i])])
        self.head = nn.Sequential(*layers)

    def forward(self, features):
        x = features.float()
        split = self.branch3(x)
        fused = torch.cat([self.branch1(x), self.branch3a(split), self.branch3b(split), self.branch_pool(x)], dim=1)
        return self.head(fused.mean((2, 3))).reshape(-1)
