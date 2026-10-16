import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def describe(frame):
    values = frame.gade.to_numpy(float)
    q1, median, q3 = np.quantile(values, [.25,.5,.75], method='linear')
    total, tied, tied_images = 0, 0, 0
    groups_with_ties = 0
    for _, part in frame.groupby('group', sort=True):
        counts = part.gade.value_counts().to_numpy()
        n = len(part)
        total += n*(n-1)//2
        count = int(sum(counts*(counts-1))//2)
        tied += count
        tied_images += int(sum(counts[counts > 1]))
        groups_with_ties += int(count > 0)
    return dict(images=len(frame), groups=frame.group.nunique(),
        score_min=float(values.min()), score_max=float(values.max()), distinct_scores=frame.gade.nunique(),
        score_mean=float(values.mean()), score_std=float(values.std(ddof=1)),
        median=float(median), q1=float(q1), q3=float(q3), iqr=float(q3-q1),
        groups_with_ties=groups_with_ties, tied_images=tied_images,
        tied_images_percent=100*tied_images/len(frame), total_within_group_pairs=total,
        tied_pairs=tied, tied_pairs_percent=100*tied/total, eligible_pairs=total-tied)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', type=Path, default=Path('data/labels.csv'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    frame = pd.read_csv(args.labels)
    if len(frame) != 3360 or frame.filename.duplicated().any() or frame.group.nunique() != 93:
        raise ValueError('Expected the complete AED label table')
    frame['category'] = frame.group.str[0].map({'a':'Sketch', 'b':'Quick Sketch', 'c':'Color Painting'})
    categories = [dict(category=name, **describe(part)) for name, part in frame.groupby('category', sort=True)]
    categories.append(dict(category='All AED', **describe(frame)))
    groups = [dict(category=part.category.iloc[0], group=name, **describe(part))
              for name, part in frame.groupby('group', sort=True)]
    args.output.mkdir(parents=True)
    pd.DataFrame(categories).to_csv(args.output / 'category_statistics.csv', index=False)
    pd.DataFrame(groups).to_csv(args.output / 'group_statistics.csv', index=False)
    print(pd.DataFrame(categories).to_string(index=False))


if __name__ == '__main__':
    main()
