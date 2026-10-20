"""Compare repository BC loss calling convention with DDP forward, without editing training source."""
import datetime
import json
import os
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.distributed as dist
import zarr
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from rl_100.model.common.normalizer import LinearNormalizer


class LossForward(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, batch):
        return self.policy.compute_loss(batch)[0]


dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=120))
rank = dist.get_rank()
torch.cuda.set_device(rank)
device = torch.device('cuda', rank)
torch.set_num_threads(2)
root = Path('/home/suchenyu/RL-100')
OmegaConf.register_new_resolver('eval', eval, replace=True)
with initialize_config_dir(str(root / 'RL-100/rl_100/config'), version_base=None):
    cfg = compose(config_name='rl100_3d_epsilon', overrides=[
        'task=adroit_door_medium', 'policy.model=skipnet', 'policy.encoder_type=dp3vib',
        'policy.use_vib=True', 'policy.use_recon=True', 'policy.beta_kl=1e-5',
        'use_action_embed=False', 'horizon=3', 'n_obs_steps=3', 'n_action_steps=1',
        'policy.scheduler_type=ddim',
    ])
data = zarr.open(str(root / 'RL-100/data/adroit_door_medium.zarr'), mode='r')['data']
norm_data = {k: np.asarray(data[src][:64], dtype=np.float32) for k, src in [
    ('action', 'action'), ('agent_pos', 'state'), ('point_cloud', 'point_cloud')]}
normalizer = LinearNormalizer()
normalizer.fit(norm_data, last_n_dims=1)
offset = rank * 6
batch = {'obs': {}, 'action': torch.tensor(norm_data['action'][offset:offset+6].reshape(2, 3, 28), device=device)}
for key in ('agent_pos', 'point_cloud'):
    a = norm_data[key][offset:offset+6]
    batch['obs'][key] = torch.tensor(a.reshape(2, 3, *a.shape[1:]), device=device)

results = {}
try:
    for mode in ('repository_module_compute_loss', 'ddp_forward_control'):
        torch.manual_seed(100 + rank)
        policy = hydra.utils.instantiate(cfg.policy)
        policy.set_normalizer(normalizer)
        policy.to(device)
        wrapped = DDP(policy if mode.startswith('repository') else LossForward(policy),
                      device_ids=[rank], find_unused_parameters=True)
        optimizer = torch.optim.SGD(wrapped.parameters(), lr=1e-3)
        distances = []
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = wrapped.module.compute_loss(batch)[0] if mode.startswith('repository') else wrapped(batch)
            loss.backward()
            optimizer.step()
            flat = torch.cat([p.detach().reshape(-1) for p in policy.parameters()])
            other = flat.clone()
            dist.broadcast(other, src=0)
            difference = (flat - other).abs().max()
            dist.all_reduce(difference, op=dist.ReduceOp.MAX)
            distances.append(difference.item())
        results[mode] = {'max_parameter_difference_after_each_update': distances}
        if rank == 0:
            print(mode, distances, flush=True)
        del optimizer, wrapped, policy, flat, other
        torch.cuda.empty_cache()
        dist.barrier()
    if rank == 0:
        results['torch_version'] = torch.__version__
        results['physical_gpus'] = os.environ.get('CUDA_VISIBLE_DEVICES')
        results['scope'] = 'Actual 3D diffusion policy, BC loss, real dataset samples; 3 updates, no checkpoints or full training'
        verification_path = root / 'rl100-setup/reports/verification.json'
        verification = json.loads(verification_path.read_text()) if verification_path.exists() else {}
        verification['ddp_diagnostic'] = results
        verification_path.write_text(json.dumps(verification, indent=2) + '\n')
finally:
    dist.destroy_process_group()
