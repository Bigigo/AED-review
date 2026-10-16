import os
os.environ['OMP_NUM_THREADS'] = '10'
import argparse
import gc
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

import common
import models
import runtime
from data import read_json, save_json, sha256, validate_cache
from references import make_bank, wrong_group
from dino_models import LAYERS


def frames_for(frame, roles, fold):
    mapping = roles[roles.outer_fold == fold].set_index('group').role
    return {role: frame[frame.group.map(mapping) == role].copy() for role in ('train', 'val', 'test')}


def train_neural(name, seed, fold, frames, args, protocol):
    runtime.configure_execution(seed)
    reference, bank, identity = (None, None, None)
    if name in ('group_mean', 'fixed_reference'):
        reference, bank, identity = make_bank(frames['train'], args.cache)
    model = models.build(name, args.assets, reference=reference).to('cuda', memory_format=torch.channels_last)
    optimizer, scheduler = runtime.optimization(model, 80)
    scaler = torch.cuda.amp.GradScaler()
    options = SimpleNamespace(seed=seed, fold=fold, workers=6, cache=args.cache, model=name,
                              image_batch=0, amp=True)
    directory = args.output / name / f'seed_{seed}' / f'fold_{fold}'
    directory.mkdir(parents=True)
    save_json(directory / 'config.json', dict(protocol, model=name, seed=seed, fold=fold,
        trainable_parameters=models.PARAMETERS[name],
        selected_blocks=list(LAYERS['gap3' if name == 'mlvs' else name]) if name in ('mlvs','gap1','gap2','gap3','gap4') else None,
        objective='RankNet' if name == 'pairwise' else 'ListNet_top_one' if name == 'listwise' else 'SmoothL1_sigmoid_midrank',
        labels_sha256=sha256(args.csv), splits_sha256=sha256(args.splits),
        reference_identity=identity))
    train_loader = runtime.loader(frames['train'], options, True)
    val_loader = runtime.loader(frames['val'], options)
    history, best_srcc, best_epoch = [], -np.inf, None
    for epoch in range(1, 81):
        model.train()
        torch.cuda.synchronize()
        start = time.perf_counter()
        losses, skipped = [], 0
        lr = optimizer.param_groups[0]['lr']
        for batch in train_loader:
            losses.append(runtime.train_group(model, batch, optimizer, scaler,
                          torch.device('cuda'), name, 0, True, False))
            skipped += int(optimizer._iaa_step_skipped)
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - start
        scheduler.step()
        groups, predictions = runtime.evaluate(model, val_loader, torch.device('cuda'), options)
        metrics = runtime.summarize_groups(groups)
        if np.isfinite(metrics['srcc']) and metrics['srcc'] > best_srcc:
            best_srcc, best_epoch = metrics['srcc'], epoch
            groups.to_csv(directory / 'val_best_groups.csv', index=False)
            predictions.to_csv(directory / 'val_best_predictions.csv', index=False)
        if epoch == 80:
            groups.to_csv(directory / 'val_final80_groups.csv', index=False)
            predictions.to_csv(directory / 'val_final80_predictions.csv', index=False)
        history.append(dict(epoch=epoch, loss=float(np.mean(losses)), lr=lr,
            lr_next=optimizer.param_groups[0]['lr'], updates=len(losses)-skipped,
            skipped_updates=skipped, grad_scale=scaler.get_scale(), best_epoch=best_epoch,
            train_seconds=train_seconds, **{'val_' + k: v for k, v in metrics.items()}))
        pd.DataFrame(history).to_csv(directory / 'history.csv', index=False)
        print(f'{name} seed={seed} fold={fold} epoch={epoch}/80 VAL_SRCC={metrics["srcc"]:.6f}', flush=True)
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    save_json(directory / 'training_complete.json', dict(epoch=80, save_weights=False,
        best_val_epoch=best_epoch, test_evaluated=False))
    del model, optimizer, scheduler, scaler, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()
    return dict(model=name, seed=seed, fold=fold, state=state, frames=frames,
                directory=directory, reference=reference, bank=bank, reference_identity=identity)


def mismatched_batches(loader, model, bank, sizes, selected):
    try:
        for batch in loader:
            group = wrong_group(batch['group'], len(batch['scores']), sizes)
            selected[batch['group']] = group
            model.reference_override = torch.from_numpy(bank[group])
            yield batch
    finally:
        model.reference_override = None


def test_neural(job, args):
    name, seed, fold = job['model'], job['seed'], job['fold']
    runtime.configure_execution(seed)
    model = models.build(name, args.assets, initialize=False, reference=job['reference']).to('cuda', memory_format=torch.channels_last)
    model.load_state_dict(job.pop('state'), strict=True)
    options = SimpleNamespace(seed=seed, fold=fold, workers=6, cache=args.cache, model=name,
                              image_batch=0, amp=True)
    output = []
    for condition in (['normal', 'mismatched'] if name == 'group_mean' else ['normal']):
        loader = runtime.loader(job['frames']['test'], options)
        selected = {}
        batches = mismatched_batches(loader, model, job['bank'],
            job['reference_identity']['group_sizes'], selected) if condition == 'mismatched' else loader
        groups, predictions = runtime.evaluate(model, batches, torch.device('cuda'), options)
        for data in (groups, predictions):
            data['model'], data['condition'], data['epoch'] = name, condition, 80
            data['checkpoint'] = 'final80'
            if selected:
                data['reference_group'] = data.group.map(selected)
        groups.to_csv(job['directory'] / f'test_final80_{condition}_groups.csv', index=False)
        predictions.to_csv(job['directory'] / f'test_final80_{condition}_predictions.csv', index=False)
        output.append(groups)
    save_json(job['directory'] / 'complete.json', dict(epoch=80, save_weights=False, test_evaluated=True))
    del model, loader
    gc.collect()
    torch.cuda.empty_cache()
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=Path, default=Path('data/labels.csv'))
    parser.add_argument('--splits', type=Path, default=Path('splits/groups.csv'))
    parser.add_argument('--image-root', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path('cache'))
    parser.add_argument('--assets', type=Path, default=Path('assets'))
    parser.add_argument('--models', nargs='+', choices=list(models.PARAMETERS) + ['ridge'], default=['mlvs'])
    parser.add_argument('--seeds', nargs='+', type=int, choices=[0,1,2], default=[0,1,2])
    parser.add_argument('--folds', nargs='+', type=int, choices=range(5), default=list(range(5)))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new output directory')
    if any(len(x) != len(set(x)) for x in (args.models, args.seeds, args.folds)):
        raise ValueError('Duplicate models, seeds or folds')
    if not torch.cuda.is_available() and any(name != 'ridge' for name in args.models):
        raise RuntimeError('The neural training protocol requires CUDA')
    frame = common.prepare_data(args.csv, args.image_root)
    frame['_cache_row'] = np.arange(len(frame))
    roles = common.load_split_roles(frame, args.splits)
    cache = validate_cache(args.cache, args.csv, frame, args.models)
    protocol = read_json(Path(__file__).parent / 'configs/protocol.json')
    args.output.mkdir(parents=True)
    save_json(args.output / 'experiment.json', dict(models=args.models, seeds=args.seeds, folds=args.folds,
        protocol=protocol, cache_identity=cache, torch=str(torch.__version__), primary='final80'))
    pending, ridge_jobs = [], []
    for name in args.models:
        if name == 'ridge':
            import ridge
            features = np.load(args.cache / 'cls.npy', mmap_mode='r')
            for fold in args.folds:
                frames = frames_for(frame, roles, fold)
                fit, selection = ridge.choose(frames['train'], frames['val'], features)
                directory = args.output / 'ridge' / f'fold_{fold}'
                directory.mkdir(parents=True)
                selection.to_csv(directory / 'validation_alpha.csv', index=False)
                save_json(directory / 'config.json', dict(alpha=fit['alpha'], fit_sha256=ridge.digest(fit),
                    independent_fits=1, seeds_are_repeat_labels=True, split_fold=fold))
                ridge_jobs.append((fold, frames, fit, directory))
            del features
        else:
            for seed in args.seeds:
                for fold in args.folds:
                    pending.append(train_neural(name, seed, fold, frames_for(frame, roles, fold), args, protocol))
    all_groups = []
    for job in pending:
        all_groups.extend(test_neural(job, args))
    if ridge_jobs:
        import ridge
        features = np.load(args.cache / 'cls.npy', mmap_mode='r')
        for fold, frames, fit, directory in ridge_jobs:
            for seed in args.seeds:
                groups, predictions = ridge.evaluate(frames['test'], features, fit, seed, fold)
                for data in (groups, predictions):
                    data['model'], data['condition'], data['epoch'] = 'ridge', 'normal', np.nan
                    data['checkpoint'] = 'val_selected_ridge'
                groups.to_csv(directory / f'seed_{seed}_test_groups.csv', index=False)
                predictions.to_csv(directory / f'seed_{seed}_test_predictions.csv', index=False)
                all_groups.append(groups)
    pd.concat(all_groups, ignore_index=True).to_csv(args.output / 'all_test_groups.csv', index=False)
    save_json(args.output / 'complete.json', dict(primary='final80', test_after_all_selected_training=True,
        save_weights=False, models=args.models, seeds=args.seeds, folds=args.folds))
    print('Completed:', args.output)


if __name__ == '__main__':
    main()
