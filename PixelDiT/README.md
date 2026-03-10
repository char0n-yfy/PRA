# PixelDiT Watermark Attack Subproject

`PixelDiT/` 目录现在承载的是一个基于 **PixelDiT-T2I** 与 **pixel mean flows (pMF)** 改写的像素空间水印攻击/受控再生成训练管线，而不是单纯的论文模型展示页。

当前实现面向以下目标：

- 使用 **PixelDiT-T2I-1B** 作为主干，在像素空间完成一步或少步再生成。
- 通过 **DINO 语义条件**、**edge 结构条件** 与可选的 **texture 分支**，在去除水印的同时尽量保留语义和结构。
- 使用 **pMF 风格的 `u/v` 双头训练目标**，并在推理阶段使用 **外置 CFG** 控制攻击强度。
- 支持 **DDP 训练**、**epoch-end eval**、**独立 eval**、**断点续训**、**TensorBoard** 与 best checkpoint 管理。

## 1. 当前子项目的核心组成

### 1.1 主干结构

当前训练模型定义在：

- `PixelDiT/models/pixeldit_t2i_pmf.py`
- `PixelDiT/models/pixeldit_blocks.py`

整体结构可以概括为：

1. Patch-level image stream
2. Semantic stream（DINO dense token + pooling）
3. Shared pixel refinement trunk
4. `u/v` branch-specific pixel refiners
5. `u/v` final output heads

其中：

- `u` 分支更直接服务最终图像重建/再生成。
- `v` 分支用于辅助速度监督。
- 推理阶段实际使用的是 `u` 分支输出构造的速度场。

### 1.2 条件分支

当前代码支持三类条件：

- `semantic`
  - 在线 DINO 编码或离线 cache
- `spatial`
  - Sobel edge 结构条件，注入 patch-level image stream
- `texture`（可选）
  - 独立 `TexNet`，生成 full dense `Δ(γ,β,α)`，当前 ablation 版只注入 `u_head_blocks`

### 1.3 训练目标

训练主逻辑在：

- `PixelDiT/train.py`
- `PixelDiT/utils/losses.py`

当前目标包含：

- `loss_u`
- `loss_v`
- `loss_u_raw`
- `loss_v_raw`
- 低噪区感知损失：`LPIPS` / `ConvNeXt`

并采用 pMF 风格的平均速度构造：

- `V = u + (t-r) * sg(du/dt)`

当前 JVP 默认使用 **finite-difference fallback**，不是强制 `torch.func.jvp`。

### 1.4 推理与评估

- 训练内 epoch-end eval：`PixelDiT/train.py`
- 独立评估入口：`PixelDiT/eval.py`
- 轻量单样本推理入口：`PixelDiT/infer.py`

注意：

- `eval.py` 是当前最完整、最可信的评估路径。
- `infer.py` 是轻量单样本工具，主要适用于基础配置；它当前**没有完整跟进 texture ablation 分支**，也不适合作为主评估入口。

## 2. 目录说明

```text
PixelDiT/
├── main.py                         # 训练入口，支持 smoke-test / resume
├── train.py                        # DDP 训练、checkpoint、epoch-end eval
├── eval.py                         # 独立完整评估入口
├── infer.py                        # 轻量单样本推理工具（功能较 eval.py 简化）
├── scripts/
│   ├── preflight_eval.py           # 训练前完整 eval 链路预检
│   └── smoke_eval.py               # 读取已有 checkpoint 做轻量 eval smoke test
├── configs/
│   ├── base_t2i_pmf.yml            # 主训练配置
│   └── base_t2i_pmf_tex_ablation.yml
│                                    # 纹理分支 ablation 配置
├── models/
│   ├── pixeldit_t2i_pmf.py         # 当前主模型
│   └── pixeldit_blocks.py          # PixelTransformerBlock / PixelFinalLayer
└── utils/
    ├── dino_encoder.py
    ├── edge.py
    ├── texture.py
    ├── losses.py
    ├── perceptual.py
    └── logging.py
```

## 3. 依赖与环境

推荐环境：

- Python 3.10+
- CUDA 可用的 PyTorch 环境
- 双卡或以上 DDP 训练环境

最小依赖建议：

```bash
pip install torch torchvision transformers timm einops pyyaml lpips tensorboard pillow numpy
```

如果你使用在线语义编码和 CLIP/ConvNeXt 评估，还需要：

- 本地可访问的 Hugging Face 模型权重，或者联网下载权限

当前默认配置里会用到这些本地模型路径：

- DINO: `/root/autodl-tmp/hf_models/dinov2-base`
- ConvNeXtV2: `/root/autodl-tmp/hf_models/convnextv2-base-22k-224`
- CLIP: `/root/autodl-tmp/hf_models/clip-vit-large-patch14`

如果你的环境不同，请修改对应 YAML。

## 4. 数据集格式

当前训练与评估默认使用 ImageNet 风格目录：

```text
/root/autodl-tmp/imagenet/
├── train/
│   ├── class_a/*.JPEG
│   ├── class_b/*.JPEG
│   └── ...
└── val/
    ├── class_a/*.JPEG
    ├── class_b/*.JPEG
    └── ...
```

实现细节：

- `train` 使用 `ImageFolder`。
- `val` 同时支持标准 `ImageFolder` 结构。
- 如果 `val` 没有 class 子目录、而是平铺图片，当前代码有 fallback 逻辑处理。

## 5. 主要配置文件

### 5.1 `PixelDiT/configs/base_t2i_pmf.yml`

这是当前主线训练配置，包含：

- 在线 DINO 语义条件
- edge 结构条件
- pMF 风格 `u/v` 训练
- epoch-end eval
- 断点续训

### 5.2 `PixelDiT/configs/base_t2i_pmf_tex_ablation.yml`

这是纹理分支实验配置，额外开启：

- 独立 `TexNet`
- 12 通道 `E_tex_raw`
- full dense `Δ(γ,β,α)`
- 仅注入 `u_head_blocks`

该配置的 `resume.enabled` 默认关闭，因为它增加了新的参数，不能直接严格复用 base run 的完整训练态 checkpoint。

## 6. 快速开始

### 6.1 只做前向形状检查

```bash
python PixelDiT/main.py --config PixelDiT/configs/base_t2i_pmf.yml --smoke-test
```

适用场景：

- 快速验证模型构建是否正常
- 不启动真正训练

### 6.2 训练前先跑一遍 eval 预检

如果你担心训练跑到 epoch 末尾才发现 eval 会炸，先运行：

```bash
python PixelDiT/scripts/preflight_eval.py \
  --config PixelDiT/configs/base_t2i_pmf.yml \
  --full-config \
  --max-samples 8 \
  --device cuda:0
```

这会：

- 生成一个随机初始化 checkpoint
- 按当前 eval 配置走完整评估链路
- 提前暴露 eval 路径问题

### 6.3 DDP 训练

基础训练命令示例：

```bash
export OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  PixelDiT/main.py --config PixelDiT/configs/base_t2i_pmf.yml
```

纹理分支 ablation：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  PixelDiT/main.py --config PixelDiT/configs/base_t2i_pmf_tex_ablation.yml
```

### 6.4 独立完整 eval

从已有 checkpoint 单独跑一次完整评估：

```bash
RUN=/root/autodl-tmp/pmf/output/PixelDiT_base_t2i_pmf/20260306-111136
CKPT=$RUN/checkpoints/checkpoint_step_00071174.pt

CUDA_VISIBLE_DEVICES=0 python PixelDiT/eval.py \
  --config $RUN/used_config.yml \
  --checkpoint $CKPT \
  --workdir $RUN \
  --global-step 71174 \
  --device cuda:0
```

说明：

- `eval.py` 会把结果写到：
  - `$RUN/eval/<timestamp>/`
- best 权重会更新到：
  - `$RUN/checkpoints/best_posthoc.pt`
  - `$RUN/checkpoints/best_regen.pt`

### 6.5 轻量 eval smoke test

如果已有 run 目录，想快速验证现有 checkpoint 的 eval 路径：

```bash
python PixelDiT/scripts/smoke_eval.py \
  --config PixelDiT/configs/base_t2i_pmf.yml \
  --workdir /path/to/run \
  --max-samples 8 \
  --device cuda:0
```

### 6.6 断点续训

当前续训字段在配置文件的 `resume:` 段，例如：

```yaml
resume:
  enabled: true
  checkpoint: /root/autodl-tmp/pmf/output/PixelDiT_base_t2i_pmf/20260306-111136/checkpoints/checkpoint_step_00071174.pt
  workdir: /root/autodl-tmp/pmf/output/PixelDiT_base_t2i_pmf/20260306-111136
  strict_model: true
  optimizer: true
  scaler: true
  rng_state: true
```

然后直接重新执行训练命令即可。

当前代码已兼容 PyTorch 2.6 的 `torch.load(..., weights_only=True)` 默认变化，训练恢复和 eval 加载完整 checkpoint 时会显式走全量反序列化。

## 7. 输出目录

默认输出目录形如：

```text
output/
└── PixelDiT_base_t2i_pmf/
    └── 20260306-111136/
        ├── used_config.yml
        ├── source_config_path.txt
        ├── tensorboard/
        │   ├── events.out.tfevents...
        │   └── train.log
        ├── checkpoints/
        │   ├── checkpoint_step_00035587.pt
        │   ├── checkpoint_step_00071174.pt
        │   ├── best_posthoc.pt
        │   ├── best_posthoc.json
        │   ├── best_regen.pt
        │   ├── best_regen.json
        │   └── ...
        ├── best_checkpoints/
        │   └── checkpoint_best_train_loss_rank1.pt
        ├── eval/
        │   ├── 20260307-002317/
        │   ├── 20260307-143012/
        │   └── ...
        └── tracebacks/
```

含义：

- `tensorboard/`
  - 训练标量、训练日志
- `checkpoints/`
  - 标准完整训练态 checkpoint
  - eval 选出的 best checkpoint
- `best_checkpoints/`
  - 仅按 train loss 排名保存的轻量 best
- `eval/<timestamp>/`
  - 每次独立 eval 或 epoch-end eval 的完整结果，互不覆盖
- `tracebacks/`
  - fatal error traceback 落盘

## 8. 关于评估与 best checkpoint

当前 eval 分两类：

- `posthoc`
  - 针对较低噪声区间的后处理重建
- `regen`
  - 针对高噪区的再生成攻击

`evaluation.sweep` 控制：

- `num_steps`
- `omega`
- `intervals`

评估完成后，代码会根据配置更新：

- `best_posthoc.pt`
- `best_regen.pt`

这些 `.pt` 文件本质上是从对应 source checkpoint 复制过来的，因此也属于完整训练态 checkpoint。

如果你要“无缝续训”，优先继续使用：

- `checkpoints/checkpoint_step_xxxxxxxx.pt`

而不是：

- `best_checkpoints/checkpoint_best_train_loss_rank1.pt`

因为后者默认不包含完整 optimizer / scaler / RNG 状态。

## 9. 当前实现中需要知道的限制

### 9.1 `infer.py` 不是最完整路径

`PixelDiT/infer.py` 当前更适合作为：

- 基础配置的单样本快速测试工具

它目前的限制包括：

- 没有完整跟进 texture branch 的建模路径
- checkpoint 加载逻辑比 `train.py` / `eval.py` 更简化
- 不适合作为主评估或论文结果生成入口

如果你要严肃比较结果，优先使用：

- `PixelDiT/eval.py`

### 9.2 训练内 epoch-end eval 强制 `num_workers=0`

这是当前故意保守设置，用来避免：

- `torchrun + Python 3.12 + multiprocessing workers`
- 在 epoch-end eval 结束后触发 semaphore / worker cleanup 崩溃

对应配置项：

- `evaluation.epoch_end_num_workers`

### 9.3 eval 目录不会自动覆盖旧结果

每次 eval 都会新建：

- `eval/<timestamp>/`

所以历史可视化和 TensorBoard 结果默认都会保留，便于比对。

## 10. 推荐工作流

如果你要稳定跑一个新实验，建议按这个顺序：

1. 先 `--smoke-test`
2. 再跑 `preflight_eval.py`
3. 确认 eval 路径正常后再开正式训练
4. 训练中断后优先从 `checkpoint_step_xxxxxxxx.pt` 续训
5. 需要严肃比较结果时，用独立 `eval.py` 跑一次完整评估

## 11. 文件级入口速查

- 训练入口：`PixelDiT/main.py`
- 训练主逻辑：`PixelDiT/train.py`
- 评估入口：`PixelDiT/eval.py`
- 单样本推理：`PixelDiT/infer.py`
- 主配置：`PixelDiT/configs/base_t2i_pmf.yml`
- 纹理分支配置：`PixelDiT/configs/base_t2i_pmf_tex_ablation.yml`
- 主模型：`PixelDiT/models/pixeldit_t2i_pmf.py`
- Pixel refiner block：`PixelDiT/models/pixeldit_blocks.py`
- edge 条件：`PixelDiT/utils/edge.py`
- texture 条件：`PixelDiT/utils/texture.py`

## 12. 当前 README 的定位

这份 README 只描述 **当前仓库里真正可运行的 PixelDiT watermark attack / regeneration 子项目**，不再试图覆盖原始论文全部模型族或泛化的 PixelDiT 展示接口。

如果你后续继续扩展：

- `texture` 分支注入位置
- `infer.py` 的功能同步
- 更完整的论文方法描述

请同步更新这份 README，避免文档再次落后于代码。
