import os
from typing import Optional, Tuple, Dict, Any
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

class NIMAScorer(nn.Module):

    def __init__(self, backbone_name: str='mobilenet_v2', pretrained: bool=True, dropout: float=0.75, freeze_backbone: bool=False, num_bins: int=10):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_bins = num_bins
        if backbone_name == 'mobilenet_v2':
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.mobilenet_v2(weights=weights)
            feat_dim = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
            self.backbone = backbone
        elif backbone_name == 'resnet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet18(weights=weights)
            feat_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone
        else:
            raise ValueError(f'Unsupported NIMA backbone: {backbone_name}')
        self.feat_dim = feat_dim
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.head = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(self.feat_dim, self.num_bins))
        self._init_head()
        anchors = torch.linspace(0.0, 1.0, steps=self.num_bins)
        self.register_buffer('score_anchors', anchors)

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                nn.init.constant_(m.bias, 0.0)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        if z.ndim > 2:
            z = torch.flatten(z, 1)
        return z

    def forward(self, x: torch.Tensor, return_aux: bool=False) -> Dict[str, Any]:
        z = self.extract_features(x)
        logits = self.head(z)
        probs = torch.softmax(logits, dim=-1)
        scores = torch.sum(probs * self.score_anchors.unsqueeze(0), dim=-1)
        out = {'scores': scores}
        if return_aux:
            out.update({'features': z, 'nima_logits': logits, 'nima_probs': probs})
        return out
