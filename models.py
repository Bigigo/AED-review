from pathlib import Path
import hashlib
import json

import torch
from torch.nn import functional as F

from dino_models import CLSScorer, LevelGAP, MeanContext
from iaa_models import registry
from losses import listnet_loss, ranknet_loss

DINO = ('mlvs', 'gap1', 'gap2', 'gap3', 'gap4', 'baseline', 'pairwise', 'listwise', 'group_mean', 'fixed_reference')
IAA = ('nima', 'mlsp', 'tanet', 'saan', 'hla_gcn')
FEATURE = dict(registry.FEATURE)
PARAMETERS = {'mlvs': 12229833, 'gap1': 9870537, 'gap2': 11050185,
              'gap3': 12229833, 'gap4': 13409481, 'baseline': 9870537,
              'pairwise': 9870537, 'listwise': 9870537,
              'group_mean': 11050185, 'fixed_reference': 11050185,
              'nima': 11181642, 'mlsp': 63813633, 'tanet': 2850544,
              'saan': 18760513, 'hla_gcn': 40334445}


def tensor_digest(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode('utf-8'))
        digest.update(str((str(value.dtype), tuple(value.shape))).encode('ascii'))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def verify_weight(assets, name):
    entries = json.loads((Path(__file__).parent / 'assets/weights.json').read_text())
    item = next(x for x in entries if x['file'] == name)
    path = Path(assets) / name
    if registry.sha256(path) == item['sha256']:
        return
    if item.get('tensor_sha256') and tensor_digest(registry.state(path)) == item['tensor_sha256']:
        return
    raise ValueError(f'Pretrained weight identity differs: {name}')


def build(name, assets, initialize=True, reference=None):
    if initialize and name in IAA:
        for file in registry.FILES[name]:
            verify_weight(assets, file)
    if name in ('mlvs', 'gap1', 'gap2', 'gap3', 'gap4'):
        model = LevelGAP('gap3' if name == 'mlvs' else name)
    elif name in ('baseline', 'pairwise', 'listwise'):
        model = CLSScorer()
    elif name in ('group_mean', 'fixed_reference'):
        if reference is None:
            raise ValueError('A TRAIN-only reference vector is required')
        model = MeanContext('mean_context' if name == 'group_mean' else name, reference)
    elif name == 'saan':
        from saan_model import SAANAED
        model = SAANAED()
        if initialize:
            source = registry.state(Path(assets) / registry.FILES['saan'][0])
            source = {k[6:] if k.startswith('model.') else k: v for k, v in source.items()}
            registry.subset_load(model.GenAes.model, source)
            registry.subset_load(model.StyAes.model, source)
            model.StyAes.vgg.model.load_state_dict(registry.state(Path(assets) / registry.FILES['saan'][1]), strict=True)
    elif name in IAA:
        model = registry.build(name, assets, initialize)
    else:
        raise ValueError(name)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if count != PARAMETERS[name]:
        raise ValueError(f'{name}: expected {PARAMETERS[name]} trainable parameters, found {count}')
    return model


def output(model, inputs, name):
    if name in DINO:
        raw = model(inputs['features']).reshape(-1).float()
        if not torch.isfinite(raw).all():
            raise RuntimeError('Nonfinite predictions')
        return raw, raw.sigmoid()
    if name == 'saan':
        raw = model(registry.imagenet(inputs['pixels']), style_stats=inputs['style']).reshape(-1).float()
        if not torch.isfinite(raw).all():
            raise RuntimeError('Nonfinite SAAN predictions')
        with torch.autocast(device_type=raw.device.type, enabled=False):
            return raw, raw.sigmoid()
    return registry.output(model, inputs, name)


def loss(raw, normalized, batch, name, device):
    if name == 'pairwise':
        return ranknet_loss(raw, batch['scores'])
    if name == 'listwise':
        return listnet_loss(raw, batch['targets'])
    return F.smooth_l1_loss(normalized, batch['targets'].to(device))
