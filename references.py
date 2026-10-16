import hashlib
import numpy as np

def digest(x):
    return hashlib.sha256(np.asarray(x,dtype=np.float32).tobytes()).hexdigest()

def make_bank(train, cache):
    rows = train['_cache_row'].to_numpy(dtype=np.int64)
    if not len(rows) or len(np.unique(rows)) != len(rows):
        raise ValueError('Empty/duplicate TRAIN rows')
    array = np.load(cache / 'cls.npy', mmap_mode='r')
    if array.ndim != 3 or array.shape[1:] != (2, 384) or array.dtype != np.float32:
        raise ValueError('Expected original float32 two-view CLS cache')
    selected = np.asarray(array[rows])
    if not np.isfinite(selected).all():
        raise ValueError('Nonfinite TRAIN features')
    global_mean = selected.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    bank = {}
    sizes = {}
    for group, part in train.groupby('group', sort=True):
        values = np.asarray(array[part['_cache_row'].to_numpy(dtype=np.int64), 0])
        bank[group] = values.mean(axis=0, dtype=np.float64).astype(np.float32)
        sizes[group] = len(part)
    identity = dict(groups=sorted(bank), images=len(rows), cache_rows=rows.tolist(), labels_used=False, query_group_used=False, global_sha256=digest(global_mean), global_definition='image-weighted TRAIN original+HFlip mean; float64 accumulation -> float32', group_definition='each TRAIN group original-view mean; float64 accumulation -> float32', group_sha256={g: digest(v) for g, v in bank.items()}, group_sizes=sizes)
    del array, selected
    return (global_mean, bank, identity)

def wrong_group(group, size, bank_sizes):
    candidates = [g for g in bank_sizes if g != group]
    if not candidates:
        raise ValueError('No different TRAIN reference group')
    prefix = group.split('_')[0]
    return min(candidates, key=lambda g: (g.split('_')[0] != prefix, abs(bank_sizes[g] - size), g))
