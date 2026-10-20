#!/usr/bin/env bash
RL100_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${RL100_ROOT}/.venv/bin/activate"
export PYTHONPATH="${RL100_ROOT}/RL-100${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_PY_MUJOCO_PATH="${RL100_ROOT}/rl100-setup/mujoco/mujoco210"
export CUDA_HOME="${RL100_ROOT}/rl100-setup/cuda"
export PATH="${CUDA_HOME}/bin:${PATH}"
export CUB_HOME="${CUDA_HOME}/include"
export CPATH="${RL100_ROOT}/rl100-setup/system-libs/usr/include${CPATH:+:${CPATH}}"
for RL100_INCLUDE in "${VIRTUAL_ENV}"/lib/python3.8/site-packages/nvidia/*/include; do
    if [ -d "${RL100_INCLUDE}" ]; then
        export CPATH="${RL100_INCLUDE}${CPATH:+:${CPATH}}"
    fi
done
unset RL100_INCLUDE
export LD_LIBRARY_PATH="${MUJOCO_PY_MUJOCO_PATH}/bin:${CUDA_HOME}/lib:${RL100_ROOT}/rl100-setup/system-libs/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export WANDB_MODE=offline
