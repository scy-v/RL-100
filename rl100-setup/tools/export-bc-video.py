"""Export one evaluation episode from a saved BC checkpoint, without touching training."""
import argparse
import json
from pathlib import Path

import dill
import hydra
import imageio
import numpy as np
import torch
from omegaconf import OmegaConf

parser = argparse.ArgumentParser()
parser.add_argument('checkpoint')
parser.add_argument('--output-dir', required=True)
args = parser.parse_args()
OmegaConf.register_new_resolver('eval', eval, replace=True)
payload = torch.load(args.checkpoint, map_location='cpu', pickle_module=dill)
cfg = payload['cfg']
epoch = dill.loads(payload['pickles']['epoch'])
out = Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)
policy = hydra.utils.instantiate(cfg.policy)
state_key = 'ema_model' if cfg.training.use_ema else 'model'
policy.load_state_dict(payload['state_dicts'][state_key])
policy.to('cuda:0').eval()
torch.manual_seed(0)
np.random.seed(0)
runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=str(out), record_video=True, eval_episodes=1)
env = runner.env
env.seed(0)
obs = env.reset()
policy.reset()
goals = 0
episode_return = 0.0
steps = 0
done = False
with torch.no_grad():
    while not done:
        inputs = {k: torch.as_tensor(obs[k], device='cuda:0', dtype=torch.float32).unsqueeze(0)
                  for k in ('point_cloud', 'agent_pos', 'image')}
        result = policy.predict_action(inputs, deterministic=True, use_cm=False)
        action = result['action'].squeeze(0).cpu().numpy()
        obs, reward, done, info = env.step(action)
        goals += int(np.sum(info['goal_achieved']))
        episode_return += float(np.sum(reward))
        steps += 1
        done = bool(np.all(done))
frames = env.env.get_video().transpose(0, 2, 3, 1)
video_path = out / f'bc-epoch-{epoch:04d}-seed-0.mp4'
with imageio.get_writer(str(video_path), fps=runner.fps, codec='libx264', macro_block_size=1) as writer:
    for frame in frames:
        writer.append_data(frame)
imageio.imwrite(out / f'bc-epoch-{epoch:04d}-preview.png', frames[len(frames)//2])
metadata = {
    'checkpoint': str(Path(args.checkpoint).resolve()), 'epoch_zero_based': epoch,
    'policy': state_key, 'video': str(video_path), 'frames': len(frames), 'fps': runner.fps,
    'steps': steps, 'return': episode_return, 'goals': goals,
    'success': goals > runner.success_threshold,
    'note': 'One separate preview episode; not a success-rate estimate.',
}
video_path.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
print(json.dumps(metadata, indent=2))
env.close()
