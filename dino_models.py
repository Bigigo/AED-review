import torch
from torch import nn

LAYERS = {"gap1": (12,), "gap2": (9,12), "gap3": (6,9,12), "gap4": (3,6,9,12)}
VARIANTS = ('mean_context', 'fixed_reference')

class DenseScore(nn.Module):

    def __init__(self):
        super().__init__()
        layers = []
        for before, after, drop in ((3072, 2048, 0.25), (2048, 1024, 0.25), (1024, 256, 0.5)):
            linear = nn.Linear(before, after)
            nn.init.kaiming_normal_(linear.weight, nonlinearity='relu')
            nn.init.zeros_(linear.bias)
            layers.extend((linear, nn.ReLU(), nn.BatchNorm1d(after, eps=0.001, momentum=0.01), nn.Dropout(drop)))
        self.embedding = nn.Sequential(*layers)
        self.score = nn.Sequential(nn.Linear(256, 64), nn.Tanh(), nn.Linear(64, 68), nn.Tanh(), nn.Linear(68, 1))

    def forward(self, x):
        return self.score(self.embedding(x)).reshape(-1).float()

class LevelGAP(nn.Module):

    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        self.input_channels = len(LAYERS[variant]) * 384
        self.readout = DenseScore()
        projection = nn.Linear(self.input_channels, 3072)
        nn.init.kaiming_normal_(projection.weight, nonlinearity='relu')
        nn.init.zeros_(projection.bias)
        self.front = nn.Sequential(projection, nn.ReLU(), nn.BatchNorm1d(3072, eps=0.001, momentum=0.01))

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:] != (1536, 5, 5) or (not len(x)):
            raise ValueError('Expected unchanged four-block patch cache [N,1536,5,5]')
        features = x[:, 1536 - self.input_channels:].float().mean((2, 3))
        return self.readout(self.front(features))

class MeanContext(nn.Module):

    def __init__(self, variant, reference):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.readout = DenseScore()
        linear = nn.Linear(768, 3072)
        nn.init.kaiming_normal_(linear.weight, nonlinearity='relu')
        nn.init.zeros_(linear.bias)
        self.front = nn.Sequential(linear, nn.ReLU(), nn.BatchNorm1d(3072, eps=0.001, momentum=0.01))
        vector = torch.as_tensor(reference, dtype=torch.float32).detach().clone()
        if vector.shape != (384,) or not torch.isfinite(vector).all():
            raise ValueError('Finite TRAIN reference [384] required')
        self.register_buffer('training_reference', vector)
        self.reference_override = None

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != 384 or (not len(x)) or (not torch.isfinite(x).all()):
            raise ValueError('Expected complete-group frozen CLS [N,384]')
        x = x.float()
        if self.reference_override is not None:
            if self.training or self.variant != 'mean_context':
                raise ValueError('Wrong-reference intervention is eval-only for mean_context')
            ref = self.reference_override.to(x).reshape(1, 384)
        elif self.variant == 'mean_context':
            ref = x.mean(dim=0, keepdim=True)
        else:
            ref = self.training_reference.to(x).reshape(1, 384)
        return self.readout(self.front(torch.cat([x, ref.expand_as(x)], dim=1)))

class CLSScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.readout = DenseScore()
        linear = nn.Linear(384, 3072)
        nn.init.kaiming_normal_(linear.weight, nonlinearity='relu')
        nn.init.zeros_(linear.bias)
        self.front = nn.Sequential(linear, nn.ReLU(), nn.BatchNorm1d(3072, eps=1e-3, momentum=.01))

    def forward(self, x):
        return self.readout(self.front(x.float()))
