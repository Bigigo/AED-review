import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr

def midrank_targets(scores):
    scores = np.asarray(scores)
    if len(scores) <= 1:
        return np.zeros(len(scores), dtype=np.float32)
    return ((rankdata(scores, method='average') - 1) / (len(scores) - 1)).astype(np.float32)

def group_metrics(scores, predictions):
    scores, predictions = (np.asarray(scores), np.asarray(predictions))
    i, j = np.triu_indices(len(scores), k=1)
    eligible = scores[i] != scores[j]
    correct = ((scores[i] - scores[j]) * (predictions[i] - predictions[j]) > 0) & eligible
    total, n_correct = (int(eligible.sum()), int(correct.sum()))
    valid = len(scores) >= 2 and np.unique(scores).size >= 2 and (np.std(predictions) > 1e-12)
    return {'group_size': len(scores), 'num_unique_scores': np.unique(scores).size, 'srcc': float(spearmanr(scores, predictions)[0]) if valid else float('nan'), 'plcc': float(pearsonr(scores, predictions)[0]) if valid else float('nan'), 'pair_acc': n_correct / total if total else float('nan'), 'eligible_pairs': total, 'correct_pairs': n_correct}

def summarize(groups):
    result = {}
    for metric in ('srcc', 'plcc', 'pair_acc'):
        result[metric] = float(groups[metric].mean())
        result[f'{metric}_valid_groups'] = int(groups[metric].notna().sum())
    return result
