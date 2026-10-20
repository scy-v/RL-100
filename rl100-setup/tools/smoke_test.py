import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def main():
    import hydra
    import metaworld
    import mujoco_py
    import open3d
    import rl_100
    import zarr
    from pytorch3d.ops import sample_farthest_points
    from rl_100.env.adroit import AdroitEnv
    from rl_100.gym_util.mjpc_diffusion_wrapper import MujocoPointcloudWrapperAdroit

    assert torch.cuda.is_available(), "CUDA is unavailable"
    value = (torch.ones(16, device="cuda") * 2).sum().item()
    assert value == 32
    points, indices = sample_farthest_points(torch.rand(1, 128, 3, device="cuda"), K=16)
    assert points.shape == (1, 16, 3)
    env = MujocoPointcloudWrapperAdroit(AdroitEnv("door", device="cuda"), "adroit_door")
    env.seed(0)
    observation = env.reset()
    assert observation["image"].shape == (3, 84, 84)
    assert observation["point_cloud"].shape == (512, 6)
    assert np.isfinite(observation["point_cloud"]).all()
    rewards = []
    for _ in range(10):
        observation, reward, done, info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32)
        )
        assert np.isfinite(reward)
        rewards.append(float(reward))
        if done:
            env.reset()
    frame = env.render("rgb_array")
    assert frame.shape == (84, 84, 3)
    assert frame.std() > 0
    output = Path(__file__).resolve().parents[1] / "reports"
    Image.fromarray(frame).save(output / "adroit-door.png")
    result = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "mujoco_py": mujoco_py.__version__,
        "open3d": open3d.__version__,
        "task": "door-v0",
        "steps": len(rewards),
        "image_shape": list(frame.shape),
        "point_cloud_shape": list(observation["point_cloud"].shape),
        "cuda_fps": "passed",
        "reward": rewards[-1],
        "status": "passed",
    }
    env.close()
    verification_path = output / "verification.json"
    verification = json.loads(verification_path.read_text()) if verification_path.exists() else {}
    verification["simulation"] = result
    verification_path.write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
