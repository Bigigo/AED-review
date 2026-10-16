import math
import torch
from torch import nn
from torchvision import models

def spatial_adjacency(height, width, ratios, device):
    yy, xx = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
    dx2 = (yy.flatten()[:, None] - yy.flatten()[None, :]).float().square()
    dy2 = (xx.flatten()[:, None] - xx.flatten()[None, :]).float().square()
    distance = (dx2[None] * ratios.reshape(-1, 1, 1).square() + dy2[None]).sqrt()
    proximity = distance.amax(dim=1).unsqueeze(2) - distance
    return proximity / (proximity.sum(dim=2, keepdim=True) + 1e-08)

class GraphConvolution(nn.Module):

    def __init__(self, dimension=1024):
        super().__init__()
        self.trans_1 = nn.Linear(dimension, dimension, bias=False)
        self.trans_2 = nn.Linear(dimension, dimension, bias=False)
        self.W = nn.Linear(dimension, dimension, bias=False)
        radius = 1 / math.sqrt(dimension)
        for layer in (self.trans_1, self.trans_2, self.W):
            nn.init.uniform_(layer.weight, -radius, radius)

    def forward(self, x, spatial=None):
        content = (self.trans_1(x) @ self.trans_2(x).transpose(1, 2)).softmax(dim=2)
        adjacency = content if spatial is None else (content + spatial) / 2
        return self.W(adjacency @ x + x)

class CoordinateGraph(nn.Module):

    def __init__(self):
        super().__init__()
        self.gcns = nn.ModuleList([GraphConvolution() for _ in range(3)])

    def forward(self, x, ratios):
        b, c, h, w = x.shape
        graph = spatial_adjacency(h, w, ratios, x.device)
        nodes = x.flatten(2).transpose(1, 2)
        for i, gcn in enumerate(self.gcns):
            nodes = gcn(nodes, graph)
            if i != len(self.gcns) - 1:
                nodes = nodes.relu()
        return nodes.transpose(1, 2).reshape(b, c, h, w)

class LatentGraph(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_state = nn.Conv2d(1024, 1024, 1)
        self.conv_proj = nn.Conv2d(1024, 25, 1)
        self.gcn = GraphConvolution()
        self.conv_extend = nn.Conv2d(1024, 1024, 1, bias=False)
        self.blocker = nn.BatchNorm2d(1024, eps=0.0001)

    def forward(self, x):
        state = self.conv_state(x).flatten(2)
        projection = self.conv_proj(x).flatten(2)
        latent = state @ projection.transpose(1, 2)
        reasoned = self.gcn(latent.transpose(1, 2))
        reconstructed = reasoned.transpose(1, 2) @ projection
        return self.blocker(self.conv_extend(reconstructed.reshape_as(x)))

class HLAGCN(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        self.backbone.fc = nn.Identity()
        self.conv1x1 = nn.Conv2d(2048, 1024, 1, bias=False)
        self.lagcn1 = CoordinateGraph()
        self.lagcn2 = LatentGraph()
        self.aesfc1 = nn.Linear(1024, 10)
        self.aesfc2 = nn.Linear(1024, 10)
        self.register_buffer('anchors', torch.linspace(0, 1, 10))
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x, ratios):
        b = self.backbone
        features = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        features = b.layer4(b.layer3(b.layer2(b.layer1(features))))
        with torch.autocast(device_type=x.device.type, enabled=False):
            features = self.conv1x1(features.float())
            spatial = self.lagcn1(features, ratios.float())
            latent = self.lagcn2(spatial)
            p1 = self.aesfc1(spatial.mean((2, 3))).softmax(dim=1)
            p2 = self.aesfc2(latent.mean((2, 3))).softmax(dim=1)
            probability = (p1 + p2) / 2
            score = probability @ self.anchors
        return {'scores': score, 'spatial_distribution': p1, 'latent_distribution': p2}
