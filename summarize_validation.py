import argparse
from pathlib import Path

import numpy as np
import pandas as pd


VARIANTS = {
    'gap1': ((12,), 9870537),
    'gap2': ((9, 12), 11050185),
    'gap3': ((6, 9, 12), 12229833),
    'gap4': ((3, 6, 9, 12), 13409481),
}
METRICS = ('srcc', 'plcc', 'pair_acc')


def aggregate(rows):
    per_run = pd.DataFrame(rows).sort_values(['variant', 'seed', 'fold']).reset_index(drop=True)
    per_seed = per_run.groupby(['variant', 'layers', 'parameters', 'seed'], as_index=False)[list(METRICS)].mean()
    output = []
    for keys, frame in per_seed.groupby(['variant', 'layers', 'parameters'], sort=False):
        variant, layers, parameters = keys
        result = dict(variant=variant, layers=layers, parameters=int(parameters), n_seeds=3, n_folds=5)
        for metric in METRICS:
            values = frame[metric].to_numpy(float)
            result[metric + '_mean'] = float(values.mean())
            result[metric + '_std'] = float(values.std(ddof=1))
        output.append(result)
    return per_run, per_seed, pd.DataFrame(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new output directory')
    rows = []
    for variant, spec in VARIANTS.items():
        layers, parameters = spec
        for seed in range(3):
            for fold in range(5):
                path = args.run / variant / ('seed_%d' % seed) / ('fold_%d' % fold) / 'history.csv'
                history = pd.read_csv(path)
                required = {'epoch', 'val_srcc', 'val_plcc', 'val_pair_acc'}
                if not required.issubset(history.columns) or history.epoch.tolist() != list(range(1, 81)):
                    raise ValueError('Incomplete validation history: %s' % path)
                final = history.iloc[-1]
                values = final[['val_srcc', 'val_plcc', 'val_pair_acc']].to_numpy(float)
                if int(final.epoch) != 80 or not np.isfinite(values).all():
                    raise ValueError('Invalid final80 validation metrics: %s' % path)
                rows.append(dict(variant=variant, layers='+'.join(map(str, layers)), parameters=parameters,
                                 seed=seed, fold=fold, srcc=values[0], plcc=values[1], pair_acc=values[2]))
    per_run, per_seed, summary = aggregate(rows)
    args.output.mkdir(parents=True)
    per_run.to_csv(args.output / 'validation_seed_fold_final80.csv', index=False)
    per_seed.to_csv(args.output / 'validation_per_seed_final80.csv', index=False)
    summary.to_csv(args.output / 'validation_summary_final80.csv', index=False)
    print('INTERNAL VALIDATION ONLY; only the 60 expected history.csv files were read.')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
