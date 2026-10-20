```bash
cd /home/suchenyu
git clone --depth 1 --filter=blob:none --no-checkout https://github.com/scy-v/RL-100.git
cd RL-100
git sparse-checkout set --no-cone --stdin < rl100-setup/config/sparse-checkout.txt
git checkout

mkdir -p /data1/suchenyu/RL-100
uv venv --python 3.8 /data1/suchenyu/RL-100/.venv
ln -s /data1/suchenyu/RL-100/.venv .venv
source .venv/bin/activate
uv pip install 'pip<24.1' setuptools==59.5.0 wheel
uv pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple -r rl100-setup/config/requirements-sim.txt torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
python -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple ninja==1.11.1.4 sapien==2.2.1 cffi==1.17.1 fasteners==0.19 glfw==2.9.0 cloudpickle==3.1.1
uv pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple -c rl100-setup/config/requirements-sim.txt -r rl100-setup/config/requirements-hardware.txt

mkdir -p rl100-setup/mujoco
curl -L https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -o /tmp/rl100-mujoco210.tar.gz
tar -xzf /tmp/rl100-mujoco210.tar.gz -C rl100-setup/mujoco

mkdir -p rl100-setup/cuda
curl -L https://conda.anaconda.org/nvidia/linux-64/cuda-nvcc-12.1.105-0.tar.bz2 -o /tmp/rl100-cuda-nvcc.tar.bz2
tar -xjf /tmp/rl100-cuda-nvcc.tar.bz2 -C rl100-setup/cuda
curl -L https://conda.anaconda.org/nvidia/linux-64/cuda-cudart-dev-12.1.105-0.tar.bz2 -o /tmp/rl100-cuda-cudart-dev.tar.bz2
tar -xjf /tmp/rl100-cuda-cudart-dev.tar.bz2 -C rl100-setup/cuda
curl -L https://conda.anaconda.org/nvidia/linux-64/cuda-cudart-12.1.105-0.tar.bz2 -o /tmp/rl100-cuda-cudart.tar.bz2
tar -xjf /tmp/rl100-cuda-cudart.tar.bz2 -C rl100-setup/cuda
curl -L https://conda.anaconda.org/nvidia/linux-64/cuda-cccl-12.1.109-0.tar.bz2 -o /tmp/rl100-cuda-cccl.tar.bz2
tar -xjf /tmp/rl100-cuda-cccl.tar.bz2 -C rl100-setup/cuda

mkdir -p rl100-setup/system-libs
cd rl100-setup/system-libs
apt-get download libvulkan1 libglew-dev libglu1-mesa-dev libegl-dev libopengl0
for package in *.deb; do dpkg-deb -x "$package" .; done
cd ../..

source rl100-setup/activate.sh
python -m pip install --no-build-isolation -e third_party/gym-0.21.0 -e third_party/mujoco-py-2.1.2.14 -e third_party/Metaworld -e third_party/rrl-dependencies/mj_envs -e third_party/rrl-dependencies/mjrl -e third_party/dexart-release -e visualizer
TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=2 python -m pip install --no-build-isolation -e third_party/pytorch3d_simplified

git clone --depth 1 https://github.com/facebookresearch/r3m.git rl100-setup/r3m
python -m pip install --no-deps --no-build-isolation -e rl100-setup/r3m
mkdir -p /data1/suchenyu/RL-100/data/pretrained/r3m/r3m_18
ln -s /data1/suchenyu/RL-100/data/pretrained/r3m /home/suchenyu/.r3m
python -m gdown 'https://drive.google.com/uc?id=1nitbHQ-GRorxc7vMUiEHjHWP5N11Jvc6' -O /home/suchenyu/.r3m/r3m_18/config.yaml
python -m gdown 'https://drive.google.com/uc?id=1A1ic-p4KtYlKXdXHcV2QV0cUzI4kn0u-' -O /home/suchenyu/.r3m/r3m_18/model.pt
python -c "import r3m; r3m.device='cpu'; r3m.load_r3m('resnet18')"
```
