import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import random
from pathlib import Path
import numpy as np
import pandas as pd
from metrics import midrank_targets, group_metrics, summarize
import torch

def prepare_data(csv_path, image_root, score_column='gade', image_ext='.jpg'):
    df = pd.read_csv(csv_path, dtype={'filename': str, 'group': str})
    required = {'filename', score_column, 'fold'}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing CSV columns: {sorted(required - set(df.columns))}')
    if df.empty or df['filename'].isna().any() or df['filename'].duplicated().any():
        raise ValueError('CSV must be nonempty with unique, nonmissing filenames.')
    df['score'] = pd.to_numeric(df[score_column], errors='raise')
    if not np.isfinite(df['score']).all():
        raise ValueError('Scores must be finite.')
    folds = pd.to_numeric(df['fold'], errors='raise')
    if not folds.isin(range(5)).all() or set(folds) != set(range(5)):
        raise ValueError('CSV must contain all five folds, numbered 0..4.')
    df['fold'] = folds.astype(int)
    if 'group' not in df.columns:

        def parse_group(name):
            parts = Path(name).stem.split('_')
            if len(parts) < 3:
                raise ValueError(f'Cannot infer group from filename: {name}')
            return '_'.join(parts[:2])
        df['group'] = df['filename'].map(parse_group)
    if df['group'].isna().any() or df['group'].str.strip().eq('').any():
        raise ValueError('Group IDs must be nonmissing.')
    if df.groupby('group')['fold'].nunique().gt(1).any():
        raise ValueError('A group appears in more than one fold.')
    root = Path(image_root).resolve()

    def image_path(name):
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Filename must be relative to image root: {name}')
        if not relative.suffix:
            relative = relative.with_suffix(image_ext)
        return str(root / relative)
    df['image_path'] = df['filename'].map(image_path)
    missing = df.loc[~df['image_path'].map(lambda p: Path(p).is_file()), 'image_path']
    if len(missing):
        raise FileNotFoundError(f'{len(missing)} images missing; first: {missing.iloc[0]}')
    if 'cache_image_path' in df.columns:
        if df['cache_image_path'].isna().any():
            raise ValueError('Missing cache_image_path in cached CSV.')
        missing_cache = df.loc[~df['cache_image_path'].map(lambda p: Path(p).is_file()), 'cache_image_path']
        if len(missing_cache):
            raise FileNotFoundError(f'Cached image missing: {missing_cache.iloc[0]}')
        df['load_path'] = df['cache_image_path']
    return df

def load_split_roles(df, path):
    roles = pd.read_csv(path, dtype={'group': str})
    required = {'outer_fold', 'group', 'role'}
    if not required.issubset(roles.columns):
        raise ValueError(f'Split file needs columns: {sorted(required)}')
    if roles['group'].isna().any() or roles.duplicated(['outer_fold', 'group']).any():
        raise ValueError('Split file has missing or duplicated group assignments.')
    if not roles['role'].isin(['train', 'val', 'test']).all():
        raise ValueError('Roles must be train, val, or test.')
    if not roles['outer_fold'].isin(range(5)).all():
        raise ValueError('Invalid outer_fold in split file.')
    all_groups = set(df['group'])
    for fold in range(5):
        fold_roles = roles[roles['outer_fold'] == fold]
        if set(fold_roles['group']) != all_groups:
            raise ValueError(f'Split groups differ from dataset in outer fold {fold}.')
        test_groups = set(fold_roles.loc[fold_roles['role'] == 'test', 'group'])
        if test_groups != set(df.loc[df['fold'] == fold, 'group']):
            raise ValueError(f'Existing test groups changed in outer fold {fold}.')
        if set(fold_roles['role']) != {'train', 'val', 'test'}:
            raise ValueError(f'Empty training, validation or test subset in fold {fold}.')
    return roles

def collate_group(batch):
    return batch[0]

def seed_worker(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    random.seed(seed)
    np.random.seed(seed)
