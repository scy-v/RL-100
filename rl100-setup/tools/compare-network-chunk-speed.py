"""Short factorial BC benchmark on real Door observations; no training files changed."""
import copy
import gc
import json
import os
import statistics
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import zarr
from omegaconf import OmegaConf
from rl_100.common.pytorch_util import dict_apply
from rl_100.model.common.normalizer import LinearNormalizer

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'rl100-setup/reports/network-chunk-speed.json'
OmegaConf.register_new_resolver('eval', eval, replace=True)
torch.set_num_threads(4)
torch.cuda.set_device(0)
B = 128
data = zarr.open(str(ROOT / 'RL-100/data/adroit_door_medium.zarr'), mode='r')['data']
arrays = {k: np.asarray(data[k][:], dtype=np.float32) for k in ['state', 'action', 'point_cloud']}
normalizer = LinearNormalizer()
normalizer.fit({'action': arrays['action'], 'agent_pos': arrays['state'], 'point_cloud': arrays['point_cloud']}, last_n_dims=1)
starts = np.arange(B) % 80 + (np.arange(B) // 80) * 100
idx = starts[:, None] + np.arange(3)[None, :]
images = np.asarray(data['img'][:int(idx.max()) + 1], dtype=np.float32)
obs = {'agent_pos': torch.from_numpy(arrays['state'][idx]),
       'point_cloud': torch.from_numpy(arrays['point_cloud'][idx]),
       'image': torch.from_numpy(images[idx])}
report = {'gpu': torch.cuda.get_device_name(0), 'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
          'batch': B, 'precision': 'FP32', 'warmup': 4, 'repeats': 3, 'steps_per_repeat': 6,
          'observation_steps': 3, 'activation': 'mish', 'rows': [],
          'scope': 'Full BC: H2D, loss forward/backward, optimizer, EMA. Head: cached synthetic condition, denoiser MSE forward/backward, optimizer, EMA. Same chunk recipe within each modality; only model, scheduler, action length vary.'}
recheck = os.environ.get('RL100_BENCH_RECHECK') == '1'
if recheck:
    report = json.loads(OUT.read_text())
    report['first_pass_rows'] = copy.deepcopy(report['rows'])

def measure(step):
    for _ in range(4): step()
    torch.cuda.synchronize()
    medians = []
    for _ in range(3):
        samples = []
        for _ in range(6):
            torch.cuda.synchronize()
            t = time.perf_counter()
            step()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - t) * 1000)
        medians.append(statistics.median(samples))
    return {'median_ms': statistics.median(medians), 'repeat_medians_ms': medians}

for kind in ['3d', '2d']:
    base = OmegaConf.load(ROOT / ('rl100-setup/reports/timing-configs/' + kind + '_flow_chunk/config.yaml'))
    for family in ['diffusion', 'flow']:
        for length in [1, 4, 16]:
            for network in ['skipnet', 'dp3']:
                if recheck and not (kind == '2d' and family == 'diffusion' and (length == 4 or (length == 1 and network == 'skipnet'))):
                    continue
                cfg = copy.deepcopy(base)
                cfg.horizon = length + 2
                cfg.n_action_steps = length
                cfg.policy.model = network
                cfg.policy.act = 'mish'
                cfg.policy.scheduler_type = 'flow' if family == 'flow' else 'ddim'
                row = {'modality': kind, 'family': family, 'network': network, 'actions': length}
                print('START', row, flush=True)
                policy = optimizer = ema = batch_gpu = None
                try:
                    torch.manual_seed(100)
                    np.random.seed(100)
                    policy = hydra.utils.instantiate(cfg.policy)
                    policy.set_normalizer(normalizer)
                    policy.cuda().train()
                    row['head_parameters'] = sum(p.numel() for p in policy.model.parameters())
                    action_idx = starts[:, None] + np.arange(length + 2)[None, :]
                    batch = {'obs': obs, 'action': torch.from_numpy(arrays['action'][action_idx])}
                    batch_gpu = dict_apply(batch, lambda x: x.cuda())
                    # Check native length support before allocating optimizer states.
                    loss = policy.compute_loss(batch_gpu)[0]
                    loss.backward()
                    policy.zero_grad(set_to_none=True)
                    del loss
                    optimizer = hydra.utils.instantiate(cfg.optimizer, params=policy.parameters())
                    ema = hydra.utils.instantiate(cfg.ema, model=copy.deepcopy(policy))
                    def full_step():
                        v = dict_apply(batch, lambda x: x.cuda())
                        loss = policy.compute_loss(v)[0]
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        ema.step(policy)
                    row['bc'] = measure(full_step)
                    del optimizer, ema
                    optimizer = ema = None
                    torch.cuda.empty_cache()
                    optimizer = hydra.utils.instantiate(cfg.optimizer, params=policy.model.parameters())
                    ema = hydra.utils.instantiate(cfg.ema, model=copy.deepcopy(policy.model))
                    x = torch.randn(B, length, 28, device='cuda')
                    condition = torch.randn(B, policy.obs_feature_dim * 3, device='cuda')
                    timestep = torch.ones(B, device='cuda', dtype=torch.long)
                    target = torch.randn_like(x)
                    def head_step():
                        prediction = policy.model(sample=x, timestep=timestep, global_cond=condition)
                        assert prediction.shape == target.shape, (prediction.shape, target.shape)
                        (prediction - target).square().mean().backward()
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        ema.step(policy.model)
                    row['head'] = measure(head_step)
                    row['status'] = 'ok'
                except RuntimeError as exc:
                    row['status'] = 'unsupported' if network == 'dp3' and length == 1 else 'error'
                    row['error'] = str(exc)
                if recheck:
                    report['rows'] = [r for r in report['rows'] if (r['modality'], r['family'], r['network'], r['actions']) != (kind, family, network, length)]
                report['rows'].append(row)
                OUT.write_text(json.dumps(report, indent=2) + '\n')
                print('RESULT', json.dumps(row), flush=True)
                policy = optimizer = ema = batch_gpu = None
                gc.collect()
                torch.cuda.empty_cache()
print('DONE', OUT, flush=True)
