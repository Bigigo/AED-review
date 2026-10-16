import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import group_metrics

METRICS = ('srcc', 'plcc', 'pair_acc')
KEYS = ['seed', 'fold', 'group']


def bootstrap(groups, metric, iterations=50000):
    rng = np.random.default_rng(20261013)
    samples = np.zeros(iterations)
    for _, part in groups.groupby('fold', sort=True):
        values = part[metric].to_numpy(float)
        for start in range(0, iterations, 1000):
            size = min(1000, iterations - start)
            samples[start:start+size] += values[rng.integers(len(values), size=(size, len(values)))].sum(axis=1)
    return np.quantile(samples / len(groups), [.025, .975])


def validate(groups, labels, splits):
    expected = labels.groupby('group', sort=True).agg(group_size=('filename', 'size'),
        num_unique_scores=('gade', 'nunique'), fold=('fold', 'first')).reset_index()
    descriptor = []
    for group, rows in labels.groupby('group', sort=True):
        count = rows.gade.value_counts().to_numpy()
        descriptor.append(dict(group=group, eligible_pairs=(len(rows)*(len(rows)-1)-sum(count*(count-1)))//2))
    expected = expected.merge(pd.DataFrame(descriptor), on='group')
    roles = splits[splits.role.eq('test')]
    if len(roles) != 93 or not expected.set_index('group').fold.equals(
            roles.set_index('group').outer_fold.reindex(expected.group)):
        raise ValueError('The label folds differ from the fixed test split')
    for model, frame in groups.groupby('model', sort=True):
        if len(frame) != 279 or set(frame.seed) != {0,1,2} or frame.duplicated(KEYS).any():
            raise ValueError(f'{model}: need three seeds with 93 unique groups each')
        for seed, part in frame.groupby('seed'):
            columns = ['group', 'fold', 'group_size', 'num_unique_scores', 'eligible_pairs']
            actual = part[columns].sort_values('group').reset_index(drop=True)
            wanted = expected[columns].sort_values('group').reset_index(drop=True)
            pd.testing.assert_frame_equal(actual, wanted, check_dtype=False)
        for metric in METRICS:
            values = frame[metric].to_numpy(float)
            finite = values[np.isfinite(values)]
            low = 0 if metric == 'pair_acc' else -1
            if np.isinf(values).any() or ((finite < low) | (finite > 1)).any():
                raise ValueError('Metric outside its valid range')
        if 'correct_pairs' in frame and not np.allclose(frame.correct_pairs / frame.eligible_pairs,
                                                       frame.pair_acc, equal_nan=True):
            raise ValueError('Pair Acc differs from its pair counts')


def from_predictions(paths, labels):
    rows = []
    frames = [pd.read_csv(path, dtype={'filename': str, 'group': str}) for path in paths]
    data = pd.concat(frames, ignore_index=True)
    if data.duplicated(['model', 'condition', 'seed', 'filename']).any():
        raise ValueError('Duplicate predictions')
    identity = data[['filename','group','fold','true_score']].merge(labels[['filename','group','fold','gade']], on='filename',
                          suffixes=('_pred','_label'), how='left', validate='many_to_one')
    if (identity.gade.isna().any() or not identity.group_pred.eq(identity.group_label).all()
            or not identity.fold_pred.eq(identity.fold_label).all()
            or not np.array_equal(identity.true_score.to_numpy(np.float32), identity.gade.to_numpy(np.float32))):
        raise ValueError('Prediction identities or true scores differ from the supplied labels')
    for keys, part in data.groupby(['model', 'condition', 'seed', 'fold', 'group'], sort=True):
        model, condition, seed, fold, group = keys
        if not np.isfinite(part[['true_score', 'pred_score']].to_numpy()).all():
            raise ValueError('Nonfinite prediction/label')
        epochs = part.epoch.dropna().unique()
        if (model == 'ridge' and len(epochs)) or (model != 'ridge' and (len(epochs) != 1 or epochs[0] != 80)):
            raise ValueError('Unexpected checkpoint epoch')
        rows.append(dict(model=model, condition=condition, seed=seed, fold=fold, group=group,
                         epoch=np.nan if model == 'ridge' else 80,
                         **group_metrics(part.true_score.to_numpy(), part.pred_score.to_numpy())))
    return pd.DataFrame(rows)


def paired(model, reference):
    result = model.merge(reference, on=KEYS, suffixes=('_model', '_reference'), validate='one_to_one')
    if len(result) != 279:
        raise ValueError('Need all 93 paired groups under three seeds')
    if not np.isfinite(result[[metric+suffix for metric in METRICS
                               for suffix in ('_model','_reference')]].to_numpy()).all():
        raise ValueError('Undefined paired metrics must not be silently omitted')
    differences = result[KEYS].copy()
    for name in ('group_size', 'num_unique_scores', 'eligible_pairs'):
        if not result[name+'_model'].eq(result[name+'_reference']).all():
            raise ValueError(f'Inconsistent paired label descriptor: {name}')
    for metric in METRICS:
        differences[metric] = result[metric+'_model'] - result[metric+'_reference']
    return differences.groupby(['fold', 'group'], sort=True)[list(METRICS)].mean().reset_index()


def main():
    parser = argparse.ArgumentParser()
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--groups', nargs='+', type=Path)
    choice.add_argument('--predictions', nargs='+', type=Path)
    parser.add_argument('--reference', default='baseline')
    parser.add_argument('--labels', type=Path, default=Path('data/labels.csv'))
    parser.add_argument('--splits', type=Path, default=Path('splits/groups.csv'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new output directory')
    labels = pd.read_csv(args.labels, dtype={'filename': str, 'group': str})
    groups = pd.concat([pd.read_csv(p, dtype={'group': str}) for p in args.groups], ignore_index=True) if args.groups else from_predictions(args.predictions, labels)
    if not (groups.epoch.eq(80) | (groups.model.eq('ridge') & groups.epoch.isna())).all():
        raise ValueError('Only final80 neural results and VAL-selected Ridge fits belong in this table')
    groups['model'] = groups.model + np.where(groups.condition.eq('normal'), '', '_' + groups.condition)
    splits = pd.read_csv(args.splits, dtype={'group': str})
    validate(groups, labels, splits)
    rows = []
    for model, part in groups.groupby('model', sort=True):
        seed_values = part.groupby('seed', sort=True)[list(METRICS)].agg(lambda x: float(np.asarray(x).mean()))
        for metric in METRICS:
            values = seed_values[metric].to_numpy(float)
            rows.append(dict(model=model, metric=metric, n_seeds=3, groups_per_seed=93,
                             mean=float(values.mean()), std=float(values.std(ddof=1))))
    args.output.mkdir(parents=True)
    groups.to_csv(args.output / 'group_metrics.csv', index=False)
    pd.DataFrame(rows).to_csv(args.output / 'performance.csv', index=False)
    if args.reference not in set(groups.model):
        raise ValueError('The reference model is absent')
    reference = groups[groups.model.eq(args.reference)]
    info = labels.groupby('group', sort=True).agg(images=('filename', 'size')).reset_index()
    info['category'] = info.group.str[0].map({'a':'Sketch', 'b':'Quick Sketch', 'c':'Color Painting'})
    comparisons = []
    for model, part in groups.groupby('model', sort=True):
        if model == args.reference:
            continue
        if not np.isfinite(part[list(METRICS)].to_numpy()).all() or not np.isfinite(reference[list(METRICS)].to_numpy()).all():
            raise ValueError('Undefined eligible group metrics retained; a complete paired CI cannot be calculated')
        differences = paired(part, reference).merge(info, on='group', validate='one_to_one')
        subsets = [('All', differences)]
        subsets += [(name, differences[differences.category.eq(name)]) for name in ('Sketch', 'Quick Sketch', 'Color Painting')]
        subsets += [('N<=30', differences[differences.images.le(30)]), ('N>30', differences[differences.images.gt(30)])]
        for subset, selected in subsets:
            for metric in METRICS:
                low, high = bootstrap(selected, metric)
                comparisons.append(dict(model=model, reference=args.reference, subset=subset,
                    metric=metric, groups=len(selected), delta=float(selected[metric].mean()),
                    ci_low=float(low), ci_high=float(high)))
    pd.DataFrame(comparisons).to_csv(args.output / 'paired_confidence_intervals.csv', index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
