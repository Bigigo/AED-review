import os
from typing import Optional, Tuple, Dict, Any
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

def load_places365_resnet18_backbone(places365_ckpt_path: Optional[str]=None, imagenet_fallback: bool=True, map_location: str='cpu') -> Tuple[nn.Module, int]:
    if places365_ckpt_path is not None:
        if not os.path.exists(places365_ckpt_path):
            raise FileNotFoundError(f'places365 checkpoint not found: {places365_ckpt_path}')
        backbone = models.resnet18(num_classes=365)
        ckpt = torch.load(places365_ckpt_path, map_location=map_location, weights_only=True)
        if isinstance(ckpt, dict):
            if 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            elif 'model_state_dict' in ckpt:
                state_dict = ckpt['model_state_dict']
            else:
                state_dict = ckpt
        else:
            raise ValueError(f'Unexpected checkpoint type: {type(ckpt)}')
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[len('module.'):]
            clean_state_dict[k] = v
        missing, unexpected = backbone.load_state_dict(clean_state_dict, strict=True)
        print(f'[Places365] Loaded checkpoint from: {places365_ckpt_path}')
        print(f'[Places365] Missing keys: {missing}')
        print(f'[Places365] Unexpected keys: {unexpected}')
        theme_dim = backbone.fc.out_features
        return (backbone, theme_dim)
    if imagenet_fallback:
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            backbone = models.resnet18(weights=weights)
        except Exception:
            backbone = models.resnet18(pretrained=True)
        theme_dim = backbone.fc.out_features
        print('[Places365] No Places365 checkpoint provided. Fallback to ImageNet-pretrained ResNet18.')
        return (backbone, theme_dim)
    raise ValueError('Neither Places365 checkpoint nor imagenet fallback is available.')

def _build_mobilenet_v2_backbone(pretrained: bool=True):
    try:
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v2(weights=weights)
    except Exception:
        backbone = models.mobilenet_v2(pretrained=pretrained)
    feat_dim = backbone.classifier[1].in_features
    backbone.classifier = nn.Identity()
    return (backbone, feat_dim)

class TANetThemeBranch(nn.Module):

    def __init__(self, pretrained: bool=True, places365_ckpt_path: Optional[str]=None, hidden_dim: int=100, out_dim: int=10, dropout: float=0.75, freeze_backbone: bool=True, use_bn: bool=True):
        super().__init__()
        if places365_ckpt_path is None and (not pretrained):
            self.backbone, theme_dim = (models.resnet18(weights=None, num_classes=365), 365)
        else:
            self.backbone, theme_dim = load_places365_resnet18_backbone(places365_ckpt_path=places365_ckpt_path, imagenet_fallback=False, map_location='cpu')
        self.theme_dim = theme_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.pre = nn.Sequential(nn.Linear(theme_dim, hidden_dim), nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity(), nn.PReLU(), nn.Dropout(dropout))
        self.hyper_w = nn.Linear(theme_dim, hidden_dim * out_dim)
        self.hyper_b = nn.Linear(theme_dim, out_dim)
        self.softmax = nn.Softmax(dim=-1)
        self._init_weights()

    def _init_weights(self):
        for module in [self.pre, self.hyper_w, self.hyper_b]:
            for m in module.modules() if isinstance(module, nn.Sequential) else [module]:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def train(self, mode=True):
        super().train(mode)
        if not any((p.requires_grad for p in self.backbone.parameters())):
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor, theme_logits=None):
        theme_logits = self.backbone(x) if theme_logits is None else theme_logits
        theme_hidden = self.pre(theme_logits)
        n = theme_logits.size(0)
        dyn_w = self.hyper_w(theme_logits).view(n, self.hidden_dim, self.out_dim)
        dyn_b = self.hyper_b(theme_logits)
        theme_vec = torch.bmm(theme_hidden.unsqueeze(1), dyn_w).squeeze(1) + dyn_b
        theme_vec = self.softmax(theme_vec)
        return {'theme_logits': theme_logits, 'theme_hidden': theme_hidden, 'theme_vec': theme_vec, 'theme_dyn_w': dyn_w, 'theme_dyn_b': dyn_b}

class TANetRGBBranch(nn.Module):

    def __init__(self, patch_size: int=12, out_dim: int=10, dropout: float=0.75):
        super().__init__()
        self.patch_size = patch_size
        self.num_tokens = patch_size * patch_size
        self.pool = nn.AdaptiveAvgPool2d((patch_size, patch_size))
        self.head = nn.Sequential(nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(self.num_tokens * self.num_tokens, out_dim), nn.Softmax(dim=-1))
        self._init_weights()

    def _init_weights(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def attention_rgb_similarity(x_rgb: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x_rgb.shape
        tokens = x_rgb.flatten(2).transpose(1, 2)
        q = F.normalize(tokens, dim=-1)
        k = F.normalize(tokens, dim=-1)
        sim = torch.bmm(q, k.transpose(1, 2))
        return sim

    def forward(self, x: torch.Tensor):
        pooled_rgb = self.pool(x)
        sim_map = self.attention_rgb_similarity(pooled_rgb)
        rgb_vec = self.head(sim_map.flatten(1))
        return {'rgb_pooled': pooled_rgb, 'rgb_sim_map': sim_map, 'rgb_vec': rgb_vec}

class TANetAestheticBranch(nn.Module):

    def __init__(self, pretrained: bool=True, out_dim: int=10, dropout: float=0.75, freeze_backbone: bool=False):
        super().__init__()
        self.backbone, feat_dim = _build_mobilenet_v2_backbone(pretrained=pretrained)
        self.feat_dim = feat_dim
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.head = nn.Sequential(nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(feat_dim, out_dim))
        self._init_weights()

    def _init_weights(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        aes_vec = self.head(feat)
        return {'aes_feat': feat, 'aes_vec': aes_vec}

class TANetAEA(nn.Module):

    def __init__(self, pretrained: bool=True, tanet_places365_ckpt_path: Optional[str]=None, tanet_branch_dim: int=10, tanet_theme_hidden_dim: int=100, tanet_rgb_patch_size: int=12, tanet_dropout: float=0.75, tanet_freeze_theme_backbone: bool=True, tanet_freeze_aes_backbone: bool=False, tanet_theme_use_bn: bool=True, tanet_output_normalized: bool=True):
        super().__init__()
        self.tanet_output_normalized = tanet_output_normalized
        self.theme_branch = TANetThemeBranch(pretrained=pretrained, places365_ckpt_path=tanet_places365_ckpt_path, hidden_dim=tanet_theme_hidden_dim, out_dim=tanet_branch_dim, dropout=tanet_dropout, freeze_backbone=tanet_freeze_theme_backbone, use_bn=tanet_theme_use_bn)
        self.rgb_branch = TANetRGBBranch(patch_size=tanet_rgb_patch_size, out_dim=tanet_branch_dim, dropout=tanet_dropout)
        self.aes_branch = TANetAestheticBranch(pretrained=pretrained, out_dim=tanet_branch_dim, dropout=tanet_dropout, freeze_backbone=tanet_freeze_aes_backbone)
        self.fusion_head = nn.Sequential(nn.ReLU(inplace=True), nn.Dropout(tanet_dropout), nn.Linear(tanet_branch_dim * 3, 1))
        self._init_weights()

    def _init_weights(self):
        for m in self.fusion_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, images: torch.Tensor, return_aux: bool=False, theme_logits=None) -> Dict[str, Any]:
        theme_out = self.theme_branch(images, theme_logits)
        rgb_out = self.rgb_branch(images)
        aes_out = self.aes_branch(images)
        fused = torch.cat([aes_out['aes_vec'], theme_out['theme_vec'], rgb_out['rgb_vec']], dim=-1)
        raw_scores = self.fusion_head(fused).squeeze(-1)
        if self.tanet_output_normalized:
            scores = torch.sigmoid(raw_scores)
        else:
            scores = raw_scores
        outputs = {'scores': scores}
        if return_aux:
            outputs.update({'features': fused, 'tanet_raw_scores': raw_scores, 'tanet_aes_feat': aes_out['aes_feat'], 'tanet_aes_vec': aes_out['aes_vec'], 'tanet_theme_logits': theme_out['theme_logits'], 'tanet_theme_hidden': theme_out['theme_hidden'], 'tanet_theme_vec': theme_out['theme_vec'], 'tanet_rgb_vec': rgb_out['rgb_vec'], 'tanet_rgb_sim_map': rgb_out['rgb_sim_map']})
        return outputs
