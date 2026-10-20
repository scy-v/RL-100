# 常用启动命令

以下命令使用当前 fork 自带的脚本配置。

## 激活环境

每次打开新终端后先执行。

```bash
cd /home/suchenyu/RL-100
source rl100-setup/activate.sh
```

## 3D Diffusion

单卡运行点云扩散策略的 BC、critic/dynamics 和离线 RL 训练。

```bash
bash scripts/Diffusion/Offline/3D/train_policy.sh rl100 adroit_door_medium 0112 100
```

## 3D Flow

单卡运行点云 Flow 策略的 BC、critic/dynamics 和离线 RL 训练。

```bash
bash scripts/Flow/Offline/3D/train_policy_flow.sh rl100 adroit_door_medium 0112 100
```

## 2D Diffusion

四卡训练图像扩散策略的 BC、critic/dynamics，再分配单卡任务运行离线 RL 参数搜索。

```bash
bash scripts/Diffusion/Offline/2D/train_policy_image_unet_two_stage.sh rl100 adroit_door_medium 0112 100 4
```

## 2D Flow

四卡训练图像 Flow 策略的 BC、critic/dynamics，再分配单卡任务运行离线 RL 参数搜索。

```bash
bash scripts/Flow/Offline/2D/train_policy_image_unet_two_stage_flow.sh rl100 adroit_door_medium 0112 100 4
```

参数依次为算法名称、任务、实验标记、随机种子；最后的 `4` 表示 GPU 数量。以上命令均从仓库根目录执行，不包含在线 RL。
