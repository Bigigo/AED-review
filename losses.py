import torch
from torch.nn import functional as F

def ranknet_loss(raw, scores):
    with torch.autocast(device_type=raw.device.type, dtype=torch.float16 if raw.is_cuda else torch.bfloat16, enabled=False):
        raw = raw.float().reshape(-1)
        scores = torch.as_tensor(scores, device=raw.device, dtype=torch.float32).reshape(-1)
        if raw.shape != scores.shape or not torch.isfinite(scores).all():
            raise ValueError('Pairwise scores and predictions must be aligned and finite')
        i, j = torch.triu_indices(len(raw), len(raw), offset=1, device=raw.device)
        sign = torch.sign(scores[i] - scores[j])
        eligible = sign != 0
        if not eligible.any():
            return raw.sum() * 0
        return F.softplus(-sign[eligible] * (raw[i[eligible]] - raw[j[eligible]])).mean()

def listnet_loss(raw, targets):
    targets = torch.as_tensor(targets, device=raw.device, dtype=torch.float32).reshape(-1)
    raw = raw.reshape(-1)
    if raw.shape != targets.shape or not len(raw):
        raise ValueError('ListNet needs one aligned, nonempty complete group')
    if not torch.isfinite(raw).all() or not torch.isfinite(targets).all() or (not ((targets >= 0) & (targets <= 1)).all()):
        raise ValueError('Finite scores and complete-group midrank targets in[0,1] required')
    with torch.autocast(device_type=raw.device.type, enabled=False):
        return -(torch.softmax(targets, dim=0) * torch.log_softmax(raw.float(), dim=0)).sum()
