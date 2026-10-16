import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def metadata_check(root):
    manifest = root / 'MANIFEST.json'
    if manifest.exists():
        for name, expected in json.loads(manifest.read_text())['files'].items():
            relative = Path(name)
            assert not relative.is_absolute() and '..' not in relative.parts
            assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected, name
    labels_path = root / 'data/labels.csv'
    roles_path = root / 'splits/groups.csv'
    assert hashlib.sha256(roles_path.read_bytes()).hexdigest() == '91943b357a5f8c83cc9b62930f9e6221e3d6dd456a9d5b20d98c32908a077be9'
    assert hashlib.sha256(labels_path.read_bytes()).hexdigest() == '9f2c1ad92a4903141442c00a7fdd0619e28e64c6b121d2db1989e346bd9f0618'
    with labels_path.open(encoding='utf-8-sig', newline='') as stream:
        labels = list(csv.DictReader(stream))
    with roles_path.open(encoding='utf-8-sig', newline='') as stream:
        roles = list(csv.DictReader(stream))
    assert len(labels) == len({r['filename'] for r in labels}) == 3360
    groups = {r['group'] for r in labels}
    assert len(groups) == 93 and len(roles) == 465
    assert Counter(r['type'] for r in labels) == {'a':1066, 'b':1045, 'c':1249}
    for fold, sizes in enumerate([(64,12,17), (63,11,19), (62,11,20), (63,11,19), (64,11,18)]):
        selected = [r for r in roles if int(r['outer_fold']) == fold]
        assert len(selected) == len({r['group'] for r in selected}) == 93
        assert {r['group'] for r in selected} == groups
        assert tuple(sum(r['role'] == name for r in selected) for name in ('train','val','test')) == sizes
        assert {r['group'] for r in selected if r['role'] == 'test'} == {r['group'] for r in labels if int(r['fold']) == fold}
    for group in groups:
        scores = [float(r['gade']) for r in labels if r['group'] == group]
        count = Counter(scores)
        tied = sum(n*(n-1)//2 for n in count.values())
        assert tied == sum(a == b for i, a in enumerate(scores) for b in scores[i+1:])
    print('Metadata: 3360 artworks, 93 groups, fixed five-fold train/validation/test roles verified')


def numerical_check(cuda=False):
    import gc
    import numpy as np
    import pandas as pd
    import torch
    import models
    import runtime
    torch.set_num_threads(2)
    from evaluate import bootstrap
    from losses import listnet_loss, ranknet_loss
    from metrics import group_metrics, midrank_targets
    assert np.allclose(midrank_targets([1,1,2,3]), [1/6,1/6,2/3,1])
    metric = group_metrics(np.array([1,1,2,3]), np.array([1,2,3,4]))
    assert metric['eligible_pairs'] == 5 and metric['pair_acc'] == 1
    assert group_metrics(np.arange(4), np.zeros(4))['pair_acc'] == 0
    assert np.isnan(group_metrics(np.arange(4), np.zeros(4))['srcc'])
    assert np.array_equal(bootstrap(pd.DataFrame({'fold':[0,0,1], 'srcc':[0.,0.,0.]}), 'srcc', 100), [0.,0.])
    for name in ('baseline','mlvs','gap1','gap2','gap4','group_mean','fixed_reference','nima','mlsp','tanet','saan','hla_gcn'):
        model = models.build(name, Path('assets'), initialize=False, reference=np.zeros(384, dtype=np.float32))
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) == models.PARAMETERS[name]
        del model
        gc.collect()
    raw = torch.tensor([-.5, .1, 1.], requires_grad=True)
    assert ranknet_loss(raw, [1,2,3]) < ranknet_loss(raw.flip(0), [1,2,3])
    assert torch.isfinite(listnet_loss(raw, [0.,.5,1.]))
    if cuda:
        assert torch.cuda.is_available()
        runtime.configure_execution(0)
        model = models.build('mlvs', Path('assets'), initialize=False).to('cuda', memory_format=torch.channels_last)
        optimizer, _ = runtime.optimization(model, 80)
        scaler = torch.cuda.amp.GradScaler()
        model.train()
        batch = dict(inputs={'features':torch.randn(6,1536,5,5)}, scores=np.arange(6, dtype=np.float32),
                     targets=torch.from_numpy(midrank_targets(np.arange(6))))
        value = runtime.train_group(model, batch, optimizer, scaler, torch.device('cuda'), 'mlvs', 0, True, False)
        assert np.isfinite(value) and not optimizer._iaa_step_skipped
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
        model.eval()
        x = batch['inputs']['features'].to('cuda')
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            before = model(x)
            after = model(x.flip(0)).flip(0)
        assert torch.allclose(before, after, atol=2e-3, rtol=2e-3)
        print('CUDA: complete-group AMP, fused AdamW, channels_last and finite gradients verified on synthetic features')
    print('Numerics: tie handling, parameter counts, ranking losses and bootstrap verified')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata-only', action='store_true')
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    metadata_check(Path(__file__).parent)
    if not args.metadata_only:
        numerical_check(args.cuda)


if __name__ == '__main__':
    main()
