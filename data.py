import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from metrics import midrank_targets
from models import DINO, FEATURE


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2 ** 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
    temporary.replace(path)


def validate_cache(root, csv_path, frame, names):
    root = Path(root)
    meta = read_json(root / 'metadata.json')
    if meta['labels_sha256'] != sha256(csv_path) or meta['filenames'] != frame.filename.tolist():
        raise ValueError('The cache and label rows differ')
    keys = set()
    for name in names:
        if name in DINO or name == 'ridge':
            keys.add('patches' if name in ('mlvs', 'gap1', 'gap2', 'gap3', 'gap4') else 'cls')
        else:
            keys.add('pixels')
            if name == 'hla_gcn':
                keys.add('ratios')
            if name in FEATURE:
                keys.add(FEATURE[name])
    for key in keys:
        item = meta['arrays'][key]
        path = root / f'{key}.npy'
        values = np.load(path, mmap_mode='r', allow_pickle=False)
        if list(values.shape) != item['shape'] or str(values.dtype) != item['dtype'] or sha256(path) != item['sha256']:
            raise ValueError(f'Invalid cache: {key}')
        if len(values) != len(frame):
            raise ValueError('Cache row count differs from labels')
    return meta


class CachedGroups(Dataset):
    def __init__(self, df, root, name, training):
        self.groups = list(df.groupby('group', sort=True))
        self.root, self.name, self.training = Path(root), name, training
        self.arrays = {}

    def __len__(self):
        return len(self.groups)

    def __getstate__(self):
        result = self.__dict__.copy()
        result['arrays'] = {}
        return result

    def array(self, key):
        if key not in self.arrays:
            self.arrays[key] = np.load(self.root / f'{key}.npy', mmap_mode='r', allow_pickle=False)
        return self.arrays[key]

    def __getitem__(self, index):
        group, rows = self.groups[index]
        indices = rows['_cache_row'].to_numpy(dtype=int)
        views = np.random.randint(0, 2, len(rows)) if self.training else np.zeros(len(rows), dtype=int)
        inputs = {}
        if self.name in DINO:
            key = 'patches' if self.name in ('mlvs', 'gap1', 'gap2', 'gap3', 'gap4') else 'cls'
            inputs['features'] = torch.from_numpy(self.array(key)[indices, views].copy())
        else:
            if self.name != 'mlsp':
                pixels = self.array('pixels')[indices].copy()
                pixels[views == 1] = pixels[views == 1, :, ::-1, :]
                inputs['pixels'] = torch.from_numpy(pixels.transpose(0, 3, 1, 2).copy())
            if self.name == 'hla_gcn':
                inputs['ratio'] = torch.from_numpy(self.array('ratios')[indices].copy())
            if self.name in FEATURE:
                key = FEATURE[self.name]
                inputs[key] = torch.from_numpy(self.array(key)[indices, views].copy())
        scores = rows.score.to_numpy(dtype='float32')
        return dict(group=group, inputs=inputs, scores=scores,
                    targets=torch.from_numpy(midrank_targets(scores)),
                    filenames=rows.filename.tolist(), image_paths=rows.filename.tolist())
