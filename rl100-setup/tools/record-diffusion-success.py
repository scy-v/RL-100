"""Record a real successful rollout from the saved online EMA diffusion policy."""
import json
import os
from pathlib import Path
import dill
import hydra
import imageio
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT=Path(__file__).resolve().parents[2]
RUN=Path('/data1/suchenyu/RL-100/data/outputs_vib_ablation/adroit_door_medium-rl100-0112_seed100/relu/dp3vib/skipnet')
OUT=ROOT/'rl100-setup/presentations/video';OUT.mkdir(parents=True,exist_ok=True)
OmegaConf.register_new_resolver('eval',eval,replace=True)
torch.set_num_threads(4)
cfg=OmegaConf.load(RUN/'.hydra/config.yaml')
policy=hydra.utils.instantiate(cfg.policy)
payload=torch.load(RUN/'checkpoints/latest.ckpt',map_location='cpu',pickle_module=dill)
sd=payload['state_dicts']['ema_model' if cfg.training.use_ema else 'model']
policy.normalizer.load_state_dict({k[len('normalizer.'):]:v for k,v in sd.items() if k.startswith('normalizer.')})
del payload,sd
checkpoint=RUN/'online_best_ema'
policy.model.load_state_dict(torch.load(checkpoint/'model.pt',map_location='cpu',weights_only=True))
policy.obs_encoder.load_state_dict(torch.load(checkpoint/'encoder.pt',map_location='cpu',weights_only=True))
policy.cuda().eval()
runner=hydra.utils.instantiate(cfg.task.env_runner,output_dir=str(OUT),record_video=False,eval_episodes=1)
env=runner.env
inner=env.env
while not hasattr(type(inner),'get_mujoco_sim') and hasattr(inner,'env'):inner=inner.env
sim=inner.get_mujoco_sim()
sim.model.vis.global_.offwidth=max(1920,sim.model.vis.global_.offwidth)
sim.model.vis.global_.offheight=max(1080,sim.model.vis.global_.offheight)

def render():
    frame=inner.get_pixels_with_width_height(1920,1080)
    if frame.shape[0]==3:frame=frame.transpose(1,2,0)
    assert frame.shape==(1080,1920,3),frame.shape
    return frame.astype(np.uint8)

trials=[]
for seed in range(3):
    torch.manual_seed(seed);np.random.seed(seed);env.seed(seed)
    obs=env.reset();policy.reset();goals=0;ret=0.;done=False;step=0
    candidate=OUT/f'3D_Diffusion_Door_seed{seed}_1080p.mp4'
    frame=render();imageio.imwrite(OUT/'preview-start.png',frame)
    # Three identical frames per control step preserve the existing 10 Hz display cadence.
    writer=imageio.get_writer(str(candidate),fps=30,codec='libx264',quality=9,macro_block_size=1,
        ffmpeg_params=['-crf','16','-pix_fmt','yuv420p'])
    with writer,torch.inference_mode():
        for _ in range(3):writer.append_data(frame)
        while not done:
            inputs={k:torch.as_tensor(obs[k],device='cuda',dtype=torch.float32).unsqueeze(0) for k in ['point_cloud','agent_pos','image']}
            action=policy.predict_action(inputs,deterministic=True,use_cm=False)['action'].squeeze(0).cpu().numpy()
            obs,reward,done,info=env.step(action)
            step+=1;goals+=int(np.sum(info['goal_achieved']));ret+=float(np.sum(reward));done=bool(np.all(done))
            frame=render()
            for _ in range(3):writer.append_data(frame)
            if step%50==0:print('ROLLOUT',seed,step,'goals',goals,flush=True)
            if goals==26:imageio.imwrite(OUT/'preview-success.png',frame)
    result={'checkpoint':str(checkpoint),'checkpoint_score':1.0,'seed':seed,'steps':step,'goals':goals,
            'success_threshold':runner.success_threshold,'success':goals>runner.success_threshold,'return':ret,
            'width':1920,'height':1080,'fps':30,'render_control_hz':10,'video':str(candidate),
            'policy':'online best EMA, DDIM; not the distilled student','gpu':os.environ.get('CUDA_VISIBLE_DEVICES')}
    trials.append(result);candidate.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    print('RESULT',json.dumps(result),flush=True)
    if result['success']:
        (OUT/'selected-video.json').write_text(json.dumps(result,indent=2)+'\n')
        break
env.close()
assert trials[-1]['success'],'No successful episode in the three recorded trials.'
