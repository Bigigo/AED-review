import hashlib
from pathlib import Path
import torch
from torch import nn
from torchvision import models
from .tanet_aed import TANetAEA
from .nima_aed import NIMAScorer
NAMES = ('baseline', 'nima', 'tanet', 'mlsp', 'saan', 'hla_gcn')
ASSETS = Path(__file__).resolve().parents[1] / 'iaa_assets'
FILES = {'baseline': ['resnet18-f37072fd.pth'], 'nima': ['resnet18-f37072fd.pth'], 'tanet': ['resnet18_places365.pth.tar', 'mobilenet_v2-b0353104.pth'], 'mlsp': ['inception_resnet_v2_tf_in1k.bin'], 'saan': ['saan_baid_pretrain.pth', 'vgg19_relu4_1_imagenet.pth'], 'hla_gcn': ['resnet50-0676ba61.pth']}
FEATURE = {'tanet': 'theme', 'saan': 'style', 'mlsp': 'mlsp'}

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2 ** 20), b''):
            h.update(block)
    return h.hexdigest()

def asset_identity(name, assets=ASSETS):
    result = {}
    for file in FILES[name]:
        path = Path(assets) / file
        if not path.is_file():
            raise FileNotFoundError(f'Offline weight missing: {path}. Extract the assets archive first.')
        result[file] = sha256(path)
    return result

def architecture_identity(name):
    files = {'baseline': [], 'nima': ['nima_aed.py'], 'tanet': ['tanet_aed.py'], 'mlsp': ['mlsp.py', 'inception_resnet_v2_offline.py'], 'saan': ['saan.py', 'saan_layers.py'], 'hla_gcn': ['hla_gcn.py']}[name]
    return {'version': 'cached_aed_v2', **{f: sha256(Path(__file__).parent / f) for f in files}}

def state(path):
    return torch.load(path, map_location='cpu', weights_only=True)

def subset_load(model, source):
    expected = model.state_dict()
    source = dict(source)
    for k, v in expected.items():
        if k.endswith('.num_batches_tracked') and k not in source:
            source[k] = torch.zeros_like(v)
    missing = [k for k, v in expected.items() if k not in source or source[k].shape != v.shape]
    if missing:
        raise ValueError(f'Pretrained backbone missing or mismatched tensors: {missing[:5]}')
    model.load_state_dict({k: source[k] for k in expected}, strict=True)

def build(name, assets=ASSETS, initialize=True):
    assets = Path(assets)
    if name not in NAMES:
        raise ValueError(name)
    if name == 'baseline':
        model = models.resnet18(weights=None)
        if initialize:
            model.load_state_dict(state(assets / FILES[name][0]), strict=True)
        linear = nn.Linear(512, 1)
        nn.init.kaiming_normal_(linear.weight, nonlinearity='linear')
        nn.init.zeros_(linear.bias)
        model.fc = nn.Sequential(nn.Dropout(0.1), linear)
    elif name == 'nima':
        model = NIMAScorer(backbone_name='resnet18', pretrained=False, dropout=0.1)
        if initialize:
            subset_load(model.backbone, state(assets / FILES[name][0]))
    elif name == 'tanet':
        model = TANetAEA(pretrained=False, tanet_places365_ckpt_path=str(assets / FILES[name][0]) if initialize else None, tanet_output_normalized=True)
        if initialize:
            subset_load(model.aes_branch.backbone, state(assets / FILES[name][1]))
    elif name == 'saan':
        from .saan import SAAN
        model = SAAN(1)
        if initialize:
            source = {k[6:] if k.startswith('model.') else k: v for k, v in state(assets / FILES[name][0]).items()}
            subset_load(model.GenAes.model, source)
            subset_load(model.StyAes.model, source)
            model.StyAes.vgg.model.load_state_dict(state(assets / FILES[name][1]), strict=True)
    elif name == 'hla_gcn':
        from .hla_gcn import HLAGCN
        model = HLAGCN()
        if initialize:
            subset_load(model.backbone, state(assets / FILES[name][0]))
    else:
        from .mlsp import MLSPHead
        model = MLSPHead()
    return model

def imagenet(pixels):
    x = pixels.float() / 255
    return (x - x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]) / x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]

def output(model, inputs, name):
    if name == 'mlsp':
        values = model(inputs['mlsp'])
    else:
        x = imagenet(inputs['pixels'])
        if name == 'tanet':
            values = model(x, theme_logits=inputs['theme'])
        elif name == 'saan':
            values = model(x, style_stats=inputs['style'])
        elif name == 'hla_gcn':
            values = model(x, inputs['ratio'])
        else:
            values = model(x)
    if isinstance(values, dict):
        values = values['scores']
    values = values.reshape(-1).float()
    if not torch.isfinite(values).all():
        raise RuntimeError(f'{name}: nonfinite predictions')
    normalized = torch.sigmoid(values) if name in ('baseline', 'mlsp') else values
    return (values, normalized)
