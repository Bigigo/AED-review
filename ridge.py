import hashlib
import numpy as np
import pandas as pd
import runtime as shared

ALPHAS = (.01,.1,1.,10.,100.,1000.)

def targets(frame):
    output = np.empty(len(frame), dtype=np.float64)
    for positions in frame.groupby('group', sort=True).indices.values():
        output[positions] = shared.shared.midrank_targets(frame.iloc[positions].score.to_numpy())
    return output

def training_design(frame, features):
    counts = frame.group.map(frame.group.value_counts()).to_numpy(dtype=float)
    values = np.asarray(features[frame['_cache_row'].to_numpy(), :, :], dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (2, 384) or (not np.isfinite(values).all()):
        raise ValueError('Existing finite original/HFlip CLS features [N,2,384] required')
    return (values.reshape(-1, 384), np.repeat(targets(frame), 2), np.repeat(1.0 / (2.0 * counts), 2))

def sufficient_statistics(x, y, weight):
    x, y, weight = (np.asarray(x, float), np.asarray(y, float), np.asarray(weight, float))
    if x.ndim != 2 or len(x) != len(y) or y.shape != weight.shape or (not len(y)):
        raise ValueError('Aligned nonempty Ridge design required')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or (not np.isfinite(weight).all()) or (weight <= 0).any():
        raise ValueError('Finite design and positive fit weights required')
    mean = np.average(x, axis=0, weights=weight)
    std = np.maximum(np.sqrt(np.average((x - mean) ** 2, axis=0, weights=weight)), 1e-08)
    intercept = float(np.average(y, weights=weight))
    z = (x - mean) / std
    return dict(mean=mean, std=std, intercept=intercept, gram=z.T @ (weight[:, None] * z), rhs=z.T @ (weight * (y - intercept)))

def solve(stats, alpha):
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('Positive finite Ridge alpha required')
    coefficient = np.linalg.solve(stats['gram'] + alpha * np.eye(len(stats['rhs'])), stats['rhs'])
    return dict(mean=stats['mean'].copy(), std=stats['std'].copy(), coefficient=coefficient, intercept=stats['intercept'], alpha=float(alpha))

def predict(x, fit):
    result = (np.asarray(x, float) - fit['mean']) / fit['std'] @ fit['coefficient'] + fit['intercept']
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite Ridge predictions')
    return result

def digest(fit):
    checksum = hashlib.sha256()
    for name in ('mean', 'std', 'coefficient', 'intercept', 'alpha'):
        checksum.update(name.encode('ascii'))
        checksum.update(np.asarray(fit[name], dtype=np.float64).tobytes())
    return checksum.hexdigest()

def evaluate(frame, features, fit, seed, fold):
    values = np.asarray(features[frame['_cache_row'].to_numpy(), 0, :], dtype=np.float64)
    scores = predict(values, fit)
    predictions = frame[['group', 'filename', 'image_path', 'score']].copy().rename(columns={'score': 'true_score'})
    predictions['seed'], predictions['fold'] = (seed, fold)
    predictions['midrank_target'] = targets(frame)
    predictions['pred_score'] = scores
    predictions['image_path'] = predictions['filename']
    predictions['score_kind'] = 'raw_linear_midrank_estimate'
    groups = []
    for group, rows in predictions.groupby('group', sort=True):
        raw = rows.pred_score.to_numpy()
        groups.append(dict(seed=seed, fold=fold, group=group, pred_min=float(raw.min()), pred_max=float(raw.max()), pred_std=float(raw.std()), unique_predictions=len(np.unique(raw)), **shared.shared.group_metrics(rows.true_score.to_numpy(), raw)))
    return (pd.DataFrame(groups), predictions)

def choose(frame_train, frame_val, features):
    x, y, weight = training_design(frame_train, features)
    stats = sufficient_statistics(x, y, weight)
    best, chosen, records = (-np.inf, None, [])
    for alpha in ALPHAS:
        fit = solve(stats, alpha)
        groups, predictions = evaluate(frame_val, features, fit, 0, -1)
        metrics = shared.summarize_groups(groups)
        eligible = bool(np.isfinite(metrics['srcc']))
        records.append(dict(alpha=alpha, selection_eligible=eligible, **metrics))
        if eligible and metrics['srcc'] > best:
            best, chosen = (metrics['srcc'], fit)
    if chosen is None:
        raise RuntimeError('All Ridge alpha candidates have undefined eligible VAL SRCC; no fallback')
    return (chosen, pd.DataFrame(records))
