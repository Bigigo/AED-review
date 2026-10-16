import os
os.environ['XFORMERS_DISABLED'] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from data import read_json, save_json, sha256
from iaa_models import registry
from iaa_models.mlsp import area_resize
from models import verify_weight
from runtime import configure_execution

BLOCKS = (2, 5, 8, 11)


def record(root, key, array, meta):
    array.flush()
    path = root / f'{key}.npy'
    shape, dtype = list(array.shape), str(array.dtype)
    array._mmap.close()
    (root / f'{key}.pending.npy').rename(path)
    meta['arrays'][key] = dict(shape=shape, dtype=dtype, sha256=sha256(path))
    save_json(root / 'metadata.json', meta)


def allocate(root, key, shape, dtype):
    path = root / f'{key}.pending.npy'
    if path.exists() or (root / f'{key}.npy').exists():
        raise FileExistsError(f'Preserve existing cache and choose another output directory: {key}')
    return np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=shape)


def pixel_cache(frame, image_root, root, meta):
    n = len(frame)
    pixels = allocate(root, 'pixels', (n, 224, 224, 3), 'uint8')
    ratios = allocate(root, 'ratios', (n,), 'float32')
    resample = getattr(Image, 'Resampling', Image).BILINEAR
    for i, row in enumerate(frame.itertuples()):
        name = Path(row.filename)
        if name.is_absolute() or '..' in name.parts:
            raise ValueError('Relative image filenames are required')
        if not name.suffix:
            name = name.with_suffix('.jpg')
        with Image.open(image_root / name) as image:
            ratios[i] = image.width / image.height
            pixels[i] = np.asarray(image.convert('RGB').resize((224, 224), resample=resample))
    record(root, 'pixels', pixels, meta)
    record(root, 'ratios', ratios, meta)


class Views(Dataset):
    def __init__(self, root):
        self.path = root / 'pixels.npy'
        self.pixels = None
        self.count = len(np.load(self.path, mmap_mode='r'))

    def __len__(self):
        return self.count

    def __getstate__(self):
        result = self.__dict__.copy()
        result['pixels'] = None
        return result

    def __getitem__(self, i):
        if self.pixels is None:
            self.pixels = np.load(self.path, mmap_mode='r')
        image = self.pixels[i]
        return i, torch.from_numpy(np.stack((image, image[:, ::-1])).transpose(0, 3, 1, 2).copy())


def check_asset(args, name):
    verify_weight(args.assets, name)


def encoder(args, key):
    if key == 'dino':
        check_asset(args, 'dinov2_vits14_pretrain.pth')
        model = torch.hub.load(str(args.dino_repo), 'dinov2_vits14', source='local', pretrained=False)
        model.load_state_dict(registry.state(args.assets / 'dinov2_vits14_pretrain.pth'), strict=True)
        if len(model.blocks) != 12 or model.embed_dim != 384 or model.patch_size != 14 or model.num_register_tokens != 0:
            raise ValueError('Expected non-register DINOv2-S/14')
    elif key == 'theme':
        for file in registry.FILES['tanet']:
            check_asset(args, file)
        model = registry.build('tanet', args.assets).theme_branch.backbone
    elif key == 'style':
        from iaa_models.saan import VGG
        check_asset(args, 'vgg19_relu4_1_imagenet.pth')
        model = VGG()
        model.model.load_state_dict(registry.state(args.assets / 'vgg19_relu4_1_imagenet.pth'), strict=True)
    elif key == 'mlsp':
        from iaa_models.mlsp import MLSPExtractor
        check_asset(args, 'inception_resnet_v2_tf_in1k.bin')
        model = MLSPExtractor()
        model.backbone.load_state_dict(registry.state(args.assets / 'inception_resnet_v2_tf_in1k.bin'), strict=True)
    else:
        raise ValueError(key)
    model = model.to('cuda', memory_format=torch.channels_last) if key == 'dino' else model.to('cuda')
    return model.eval().requires_grad_(False)


@torch.no_grad()
def features(args, frame, meta, key):
    if key == 'dino':
        configure_execution(0)
    else:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        torch.use_deterministic_algorithms(True)
    model = encoder(args, key)
    n = len(frame)
    shapes = {'cls': (n, 2, 384), 'patches': (n, 2, 1536, 5, 5)} if key == 'dino' else {
        key: (n, 2, *{'theme': (365,), 'style': (512, 2), 'mlsp': (16928, 5, 5)}[key])}
    arrays = {name: allocate(args.cache, name, shape, 'float16' if name == 'mlsp' else 'float32')
              for name, shape in shapes.items()}
    batch_size = args.batch_size if key == 'dino' else args.batch_size // 2
    if batch_size < 1 or (key != 'dino' and args.batch_size % 2):
        raise ValueError('IAA extraction batch size counts views and must be a positive even number')
    batches = DataLoader(Views(args.cache), batch_size=batch_size, shuffle=False,
                         num_workers=args.workers, prefetch_factor=2 if args.workers else None,
                         persistent_workers=args.workers > 0, pin_memory=True)
    for indices, images in batches:
        count = len(indices)
        layout = torch.channels_last if key == 'dino' else torch.contiguous_format
        images = images.reshape(-1, 3, 224, 224).to('cuda', non_blocking=True, memory_format=layout)
        if key == 'dino':
            with torch.autocast('cuda', dtype=torch.float16):
                intermediate = model.get_intermediate_layers(registry.imagenet(images), n=list(BLOCKS),
                    reshape=True, return_class_token=True, norm=True)
            if any(p.shape[1:] != (384, 16, 16) or c.shape[1:] != (384,) for p, c in intermediate):
                raise ValueError('Unexpected DINOv2 feature dimensions')
            values = {'cls': intermediate[-1][1].float(),
                      'patches': torch.cat([area_resize(p.float(), 5) for p, _ in intermediate], dim=1)}
        else:
            x = images.float() / 127.5 - 1 if key == 'mlsp' else registry.imagenet(images)
            output = model(x)
            if key == 'style':
                flat = output.flatten(2)
                output = torch.stack((flat.mean(2), flat.std(2) + 1e-6), dim=-1)
            values = {key: output}
        for name, output in values.items():
            converted = output.float().cpu().numpy().reshape(count, 2, *shapes[name][2:]).astype(arrays[name].dtype)
            if not np.isfinite(converted).all():
                raise ValueError(f'Nonfinite {name} cache')
            arrays[name][indices.numpy()] = converted
    for name, values in arrays.items():
        record(args.cache, name, values, meta)
    meta.setdefault('encoders', {})[key] = dict(
        weights={file: sha256(args.assets / file) for file in (
            ['dinov2_vits14_pretrain.pth'] if key == 'dino' else registry.FILES[{'theme': 'tanet', 'style': 'saan', 'mlsp': 'mlsp'}[key]])},
        extraction_precision='CUDA float16 AMP' if key == 'dino' else 'float32',
        allow_tf32=key == 'dino', channels_last=key == 'dino',
        torch=str(torch.__version__))
    save_json(args.cache / 'metadata.json', meta)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=Path, default=Path('data/labels.csv'))
    parser.add_argument('--image-root', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path('cache'))
    parser.add_argument('--assets', type=Path, default=Path('assets'))
    parser.add_argument('--dino-repo', type=Path, default=Path('assets/dinov2_repo'))
    parser.add_argument('--features', nargs='*', choices=['dino', 'theme', 'style', 'mlsp'], default=['dino'])
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError('Positive batch size and nonnegative workers required')
    if not torch.cuda.is_available() and args.features:
        raise RuntimeError('Feature preparation uses the reported CUDA extraction protocol')
    frame = pd.read_csv(args.csv, dtype={'filename': str})
    if frame.filename.duplicated().any() or len(frame) != 3360:
        raise ValueError('Expected 3360 unique AED filenames')
    args.cache.mkdir(parents=True, exist_ok=True)
    metadata = args.cache / 'metadata.json'
    if metadata.exists():
        meta = read_json(metadata)
        if meta['labels_sha256'] != sha256(args.csv) or meta['filenames'] != frame.filename.tolist():
            raise ValueError('Existing cache belongs to another label table')
        for name, item in meta['arrays'].items():
            if sha256(args.cache / f'{name}.npy') != item['sha256']:
                raise ValueError(f'Cache checksum mismatch: {name}')
    else:
        meta = dict(labels_sha256=sha256(args.csv), filenames=frame.filename.tolist(), arrays={},
                    preprocessing='RGB; PIL bilinear direct resize 224x224; original/HFlip views; ImageNet normalization for DINO, NIMA, TANet, SAAN and HLA-GCN; [-1,1] for MLSP')
        pixel_cache(frame, args.image_root, args.cache, meta)
    for key in args.features:
        names = ['cls', 'patches'] if key == 'dino' else [key]
        if not all(name in meta['arrays'] for name in names):
            features(args, frame, meta, key)
    print('Cache preparation complete')


if __name__ == '__main__':
    main()
