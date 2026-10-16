import os
os.environ['OMP_NUM_THREADS'] = '10'
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import common as shared
import models as registry
from data import CachedGroups
def optimization(model, epochs):
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=0.0001, weight_decay=0.0001, fused=bool(parameters and parameters[0].is_cuda))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-06)
    return (optimizer, scheduler)

def configure_execution(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    torch.use_deterministic_algorithms(False)

def chunks(n, size):
    size = size or n
    bounds = list(range(0, n, size)) + [n]
    if len(bounds) > 2 and bounds[-1] - bounds[-2] == 1:
        del bounds[-2]
    return [slice(a, b) for a, b in zip(bounds[:-1], bounds[1:])]

def input_slice(inputs, part, device):
    result = {k: v[part].to(device, non_blocking=True) for k, v in inputs.items()}
    if device.type == 'cuda' and 'pixels' in result:
        result['pixels'] = result['pixels'].contiguous(memory_format=torch.channels_last)
    return result

def train_group(model, batch, optimizer, scaler, device, name, image_batch, amp, offload_activations=False):
    if image_batch != 0 or offload_activations:
        raise ValueError('Training requires a complete group without activation offload.')
    if scaler.is_enabled() != bool(amp and device.type == 'cuda'):
        raise ValueError('GradScaler and CUDA autocast must use the same AMP setting.')
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16, enabled=amp and device.type == 'cuda'):
        _, normalized = registry.output(model, input_slice(batch['inputs'], slice(None), device), name)
        loss = registry.loss(_, normalized, batch, name, device)
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite training loss')
    scale_before = scaler.get_scale()
    if scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        finite = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
        if not finite or not torch.stack(finite).all():
            raise RuntimeError('Nonfinite or missing gradients')
        optimizer.step()
    optimizer._iaa_step_skipped = scaler.is_enabled() and scaler.get_scale() < scale_before
    return loss.detach().item()

def loader(frame, args, training=False):
    options = dict(batch_size=1, shuffle=training, num_workers=args.workers, pin_memory=torch.cuda.is_available(), collate_fn=shared.collate_group, worker_init_fn=shared.seed_worker, generator=torch.Generator().manual_seed(args.seed))
    if args.workers:
        options.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(CachedGroups(frame, args.cache, args.model, training), **options)

@torch.no_grad()
def evaluate(model, batches, device, args):
    model.eval()
    groups, predictions = ([], [])
    for batch in batches:
        raw, norm = ([], [])
        for part in chunks(len(batch['targets']), args.image_batch):
            with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16, enabled=args.amp and device.type == 'cuda'):
                r, s = registry.output(model, input_slice(batch['inputs'], part, device), args.model)
            raw.append(r.cpu().numpy())
            norm.append(s.cpu().numpy())
        raw, norm = (np.concatenate(raw), np.concatenate(norm))
        groups.append({'seed': args.seed, 'fold': args.fold, 'group': batch['group'], 'pred_min': float(raw.min()), 'pred_max': float(raw.max()), 'pred_std': float(np.std(raw)), 'unique_predictions': int(np.unique(raw).size), **shared.group_metrics(batch['scores'], raw)})
        for i, file in enumerate(batch['filenames']):
            predictions.append({'seed': args.seed, 'fold': args.fold, 'group': batch['group'], 'filename': file, 'image_path': batch['filenames'][i], 'true_score': float(batch['scores'][i]), 'midrank_target': float(batch['targets'][i]), 'pred_score': float(raw[i]), 'pred_normalized': float(norm[i])})
    return (pd.DataFrame(groups), pd.DataFrame(predictions))

def summarize_groups(groups):
    result = shared.summarize(groups)
    eligible = groups.group_size.ge(2) & groups.num_unique_scores.gt(1)
    count = int(eligible.sum())
    for metric in ('srcc', 'plcc', 'pair_acc'):
        valid = int(groups.loc[eligible, metric].notna().sum())
        result[metric + '_valid_group_mean'] = result[metric]
        result[metric + '_eligible_groups'] = count
        result[metric + '_valid_groups'] = valid
        if valid != count or count == 0:
            result[metric] = float('nan')
    return result
