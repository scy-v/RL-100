"""Single-GPU matched-batch BC benchmark; never writes training checkpoints."""
import argparse
import copy
import gc
import json
import statistics
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import hydra
import numpy as np
import torch
import zarr
from omegaconf import OmegaConf
from rl_100.model.common.normalizer import LinearNormalizer
from rl_100.common.pytorch_util import dict_apply

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--steps', type=int, default=20)
parser.add_argument('--output', default='rl100-setup/reports/diffusion-speed.json')
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
OmegaConf.register_new_resolver('eval', eval, replace=True)
torch.set_num_threads(4)
torch.manual_seed(100)
np.random.seed(100)
torch.cuda.set_device(0)
assert torch.cuda.is_available()

# Identical, real Door sequences for both policies; avoid episode boundaries.
data = zarr.open(str(root / 'RL-100/data/adroit_door_medium.zarr'), mode='r')['data']
B = args.batch_size
starts = np.arange(B) % 96 + (np.arange(B) // 96) * 100
indices = starts[:, None] + np.arange(3)[None, :]
arrays = {k: np.asarray(data[k][:], dtype=np.float32) for k in ['state','action','point_cloud']}
normalizer = LinearNormalizer()
normalizer.fit({'action': arrays['action'], 'agent_pos': arrays['state'], 'point_cloud': arrays['point_cloud']}, last_n_dims=1)
images = np.asarray(data['img'][:int(indices.max())+1], dtype=np.float32)
cpu_batch = {'obs': {
    'agent_pos': torch.from_numpy(arrays['state'][indices]),
    'point_cloud': torch.from_numpy(arrays['point_cloud'][indices]),
    'image': torch.from_numpy(images[indices]),
}, 'action': torch.from_numpy(arrays['action'][indices])}
del arrays, images

class Timings:
    def __init__(self):
        self.active = False
        self.events = []
        self.gpu = defaultdict(list)
        self.wall = defaultdict(list)
    @contextmanager
    def region(self, key):
        if not self.active:
            yield
            return
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t = time.perf_counter()
        start.record()
        try:
            yield
        finally:
            end.record()
            self.events.append((key, start, end, (time.perf_counter()-t)*1000))
    def flush(self):
        torch.cuda.synchronize()
        groups, wall = defaultdict(float), defaultdict(float)
        for key, start, end, wt in self.events:
            groups[key] += start.elapsed_time(end)
            wall[key] += wt
        for key, value in groups.items(): self.gpu[key].append(value)
        for key, value in wall.items(): self.wall[key].append(value)
        self.events.clear()
    def result(self):
        return {key: {'gpu_elapsed_ms_median': statistics.median(value),
                      'wall_ms_median': statistics.median(self.wall[key])}
                for key, value in self.gpu.items()}

def wrap_method(obj, name, timer, label):
    original = getattr(obj, name)
    def wrapped(*a, **kw):
        with timer.region(label):
            return original(*a, **kw)
    setattr(obj, name, wrapped)

paths = {
    '3d': 'outputs_vib_ablation/adroit_door_medium-rl100-timing_20261020_seed100/relu/dp3vib/skipnet',
    '2d': 'outputs_2d/adroit_door_medium-rl100-timing_20261020_seed100/mish/resnet/skipnet',
}
results = {'device': torch.cuda.get_device_name(0), 'physical_gpu': __import__('os').environ.get('CUDA_VISIBLE_DEVICES'),
           'torch': torch.__version__, 'batch_size': B, 'obs_steps': 3, 'dtype': 'float32',
           'cuda_launch_blocking': __import__('os').environ.get('CUDA_LAUNCH_BLOCKING'),
           'steps_per_pass': args.steps, 'warmup_steps': 5,
           'scope': 'BC training only; matched input batch; no DDP, dataloader workers, environment rollout, checkpoint I/O or LR-scheduler timing. Original saved author-launcher policy and optimizer settings. CUDA events report elapsed stream time, including host gaps; nested regions are inclusive.',
           'models': {}}

for kind in ['3d','2d']:
    print('START', kind, flush=True)
    cfg = OmegaConf.load(root/'RL-100/data'/paths[kind]/'.hydra/config.yaml')
    torch.manual_seed(100)
    policy = hydra.utils.instantiate(cfg.policy)
    policy.set_normalizer(normalizer)
    policy.cuda().train()
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=policy.parameters())
    ema = hydra.utils.instantiate(cfg.ema, model=copy.deepcopy(policy))
    timer = Timings()
    encoder = policy.obs_encoder
    split = {'enabled': False}
    original_recon = encoder.Recon_VIB_loss
    def encoder_call(*a, **kw):
        with timer.region('observation_total'):
            aux, items, features = original_recon(*a, **kw)
        split['feature_shape'] = list(features.shape)
        if split['enabled']:
            leaf = features.detach().requires_grad_(True)
            split.update(aux=aux, features=features, leaf=leaf)
            return aux.detach(), items, leaf
        return aux, items, features
    encoder.Recon_VIB_loss = encoder_call
    wrap_method(policy.model, 'forward', timer, 'denoiser_forward')
    if kind == '2d':
        wrap_method(policy, '_augment_rgb_obs', timer, 'augmentation')
        wrap_method(encoder, '_apply_spatial_transform', timer, 'spatial_preprocessing')
        wrap_method(encoder, '_vib_forward', timer, 'vib_heads_sampling_kl')
        for m in encoder.key_model_map.values(): wrap_method(m, 'forward', timer, 'backbone_forward')
        for m in encoder.key_normalize_map.values(): wrap_method(m, 'forward', timer, 'imagenet_normalization')
        for m in encoder.decoders.values(): wrap_method(m, 'forward', timer, 'reconstruction_decoder')
    else:
        wrap_method(encoder.extractor, 'forward', timer, 'backbone_forward')
        wrap_method(encoder.state_mlp, 'forward', timer, 'state_encoder_forward')
        for name in ['pc_decoder','state_decoder']:
            wrap_method(getattr(encoder, name), 'forward', timer, 'reconstruction_decoder')
        for name in ['latent_mu_pc','latent_logvar_pc','latent_mu_state','latent_logvar_state']:
            wrap_method(getattr(encoder,name), 'forward', timer, 'vib_linear_heads')
        # chamfer_distance is the 3D reconstruction metric, beyond its decoder.
        import rl_100.model.vision.pointnet_extractor as pe
        original_chamfer = pe.chamfer_distance
        def timed_chamfer(*a, **kw):
            with timer.region('reconstruction_chamfer_metric'):
                return original_chamfer(*a, **kw)
        pe.chamfer_distance = timed_chamfer

    gpu_batch = dict_apply(cpu_batch, lambda x: x.cuda())
    # Verify that splitting backward at the encoder feature boundary preserves gradients.
    def seed(): torch.manual_seed(777); np.random.seed(777)
    seed(); optimizer.zero_grad(set_to_none=True)
    canonical_loss = policy.compute_loss(gpu_batch)[0]
    canonical_loss.backward()
    canonical_grad = {n: p.grad.detach().clone() for n,p in policy.named_parameters() if p.grad is not None}
    optimizer.zero_grad(set_to_none=True); split['enabled'] = True; seed()
    split_loss = policy.compute_loss(gpu_batch)[0]
    split_loss.backward()
    torch.autograd.backward([split['features'],split['aux']], [split['leaf'].grad, torch.ones_like(split['aux'])])
    max_abs = max((p.grad-canonical_grad[n]).abs().max().item() for n,p in policy.named_parameters() if n in canonical_grad)
    max_ref = max(x.abs().max().item() for x in canonical_grad.values())
    assert abs(canonical_loss.item()-split_loss.item()) < 1e-5
    assert max_abs <= max(1e-5, max_ref*1e-4), (kind,max_abs,max_ref)
    verification={'loss_difference':abs(canonical_loss.item()-split_loss.item()),'gradient_max_absolute_difference':max_abs,'reference_max_abs_gradient':max_ref}
    split['enabled']=False
    del canonical_grad, canonical_loss, split_loss
    optimizer.zero_grad(set_to_none=True)

    def step(split_backward=False, transfer=True):
        split['enabled'] = split_backward
        with timer.region('total_step'):
            with timer.region('host_to_device'):
                batch = dict_apply(cpu_batch, lambda x: x.cuda()) if transfer else gpu_batch
            with timer.region('loss_forward'):
                loss = policy.compute_loss(batch)[0]
            with timer.region('denoiser_backward' if split_backward else 'backward_total'):
                loss.backward()
            if split_backward:
                with timer.region('observation_backward'):
                    torch.autograd.backward([split['features'],split['aux']], [split['leaf'].grad,torch.ones_like(split['aux'])])
            with timer.region('optimizer'):
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            with timer.region('ema'):
                ema.step(policy)
        return loss.item()
    for _ in range(5): step()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    # Baseline without nested instrumentation; wall latency measured separately.
    full_wall=[]
    for _ in range(args.steps):
        t=time.perf_counter(); step(); torch.cuda.synchronize(); full_wall.append((time.perf_counter()-t)*1000)
    baseline_wall=statistics.median(full_wall)
    peak=torch.cuda.max_memory_allocated()/2**20
    # Detailed split pass with event ranges; original modules and losses retained.
    timer.active=True
    for _ in range(args.steps):
        step(split_backward=True);timer.flush()
    measured=timer.result();timer.active=False
    # Matched-batch ablation: remove only RGB augmentation to quantify its contribution.
    no_aug=None
    if kind=='2d':
        original_aug=policy.use_aug;policy.use_aug=False
        for _ in range(3):step()
        values=[]
        for _ in range(args.steps):
            t=time.perf_counter();step();torch.cuda.synchronize();values.append((time.perf_counter()-t)*1000)
        no_aug=statistics.median(values);policy.use_aug=original_aug
    record={'baseline_step_wall_ms_median':baseline_wall,'samples_per_second':B/(baseline_wall/1000),
            'peak_allocated_mib':peak,'parameters_total':sum(p.numel() for p in policy.parameters()),
            'parameters_trainable':sum(p.numel() for p in policy.parameters() if p.requires_grad),
            'encoder_parameters':sum(p.numel() for p in encoder.parameters()),
            'denoiser_parameters':sum(p.numel() for p in policy.model.parameters()),
            'observation_feature_shape':split['feature_shape'],
            'conditioning_width':policy.obs_feature_dim*policy.n_obs_steps,
            'encoder_type':policy.encoder_type,'policy_model':cfg.policy.model,'policy_activation':cfg.policy.act,
            'use_aug':getattr(policy,'use_aug',False),'gradient_split_check':verification,
            'module_timings':measured,'no_augmentation_step_wall_ms_median':no_aug}
    results['models'][kind]=record
    print('RESULT',kind,json.dumps(record),flush=True)
    Path(root/args.output).write_text(json.dumps(results,indent=2)+'\n')
    if kind=='3d': pe.chamfer_distance=original_chamfer
    del policy,encoder,optimizer,ema,gpu_batch
    split.clear();gc.collect();torch.cuda.empty_cache()
print('DONE',root/args.output,flush=True)
