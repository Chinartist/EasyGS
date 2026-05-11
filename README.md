# EasyGS

EasyGS 是一个围绕 3D Gaussian Splatting 的轻量训练框架。它保留了简单的 `GSer(...)` Python 入口，同时提供：

- 从 COLMAP 数据训练 3DGS
- 从内存中的相机矩阵、图像、点云训练 3DGS
- 加载已有 `.ply` 或 checkpoint 做评估/渲染
- 固定 Gaussian 后优化相机参数
- 只拟合 SH 颜色
- CLI 命令行工作流
- callback 扩展机制

这份 README 按“从零跑起来”的顺序写。第一次用可以从上往下照着走。

## 1. 项目结构

核心代码已经拆成几个职责清晰的模块：

```text
EasyGS/
  Trainer.py                 # 训练调度：采样相机、render、loss、optimizer、densify
  config.py                  # dataclass 配置对象
  data.py                    # COLMAP / arrays 场景构建
  losses.py                  # 训练 loss 和 eval metrics
  callbacks.py               # progress / eval / checkpoint / wandb hooks
  checkpoint.py              # checkpoint 保存和预训练加载
  outputs.py                 # 渲染图、深度、法线、alpha、extra attrs 输出
  colmap_io.py               # COLMAP text 格式导出
  cli.py                     # easygs 命令行入口
  scene/                     # Camera / GaussianModel / COLMAP 读取
  gaussian_renderer/         # CUDA rasterizer 调用封装
  submodules/                # diff-gaussian-rasterization / simple-knn
```

扩展时优先改对应模块：

- 加新数据源：改 `data.py`
- 加新 loss：改 `losses.py`
- 改保存内容：改 `checkpoint.py`
- 改渲染输出：改 `outputs.py`
- 加训练 hook：写 `callbacks.py` callback

## 2. 环境要求

EasyGS 依赖 CUDA 版 PyTorch 和两个 CUDA 扩展。建议环境：

- Python >= 3.9
- NVIDIA GPU
- CUDA toolkit 与 PyTorch CUDA 版本匹配
- C++/CUDA 编译工具链
- COLMAP 格式数据

安装 PyTorch 时，请按你的 CUDA 版本选择官方命令。例如 CUDA 12.1 环境通常类似：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

如果你的 CUDA 版本不同，请到 PyTorch 官网复制对应安装命令。

## 3. 安装

进入项目根目录：

```bash
cd /path/to/EasyGS
```

安装 EasyGS 本体：

```bash
pip install -e .
```

安装 CUDA 扩展：

```bash
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

如果需要 wandb 记录实验：

```bash
pip install -e ".[logging]"
```

验证命令行入口是否可用：

```bash
easygs --help
```

常见安装问题：

- `No module named diff_gauss_pose`：`submodules/diff-gaussian-rasterization` 没装成功。
- `No module named simple_knn._C`：`submodules/simple-knn` 没装成功。
- CUDA 编译失败：检查 CUDA toolkit、PyTorch CUDA 版本、编译器是否匹配。

## 4. 准备 COLMAP 数据

最常见的数据目录应长这样：

```text
/path/to/scene/
  images/
    000001.png
    000002.png
    ...
  sparse/
    0/
      cameras.bin 或 cameras.txt
      images.bin 或 images.txt
      points3D.bin 或 points3D.txt
```

EasyGS 会优先读二进制 COLMAP 文件：

```text
sparse/0/images.bin
sparse/0/cameras.bin
sparse/0/points3D.bin
```

如果 `.bin` 不存在，会尝试读 `.txt`：

```text
sparse/0/images.txt
sparse/0/cameras.txt
sparse/0/points3D.txt
```

可选监督数据目录：

```text
/path/to/scene/
  depths/       # .npy, 文件名和 image 对应
  normals/      # .npy, 文件名和 image 对应
  skymask/      # .png, 文件名和 image 对应
  extra_attrs/  # .npy, 文件名和 image 对应
```

例如图像是：

```text
images/000001.png
```

对应深度文件会按名字推断为：

```text
depths/000001.npy
```

## 5. 最快训练：CLI

最小命令：

```bash
easygs train-colmap /path/to/scene --save-dir /path/to/scene/3dgs
```

常用训练命令：

```bash
easygs train-colmap /path/to/scene \
  --save-dir /path/to/scene/3dgs \
  --iterations 50000 \
  --save-interval 10000 \
  --eval-interval 10000 \
  --eval-rate 0.1 \
  --height 520 \
  --width 930
```

带 alpha/skymask：

```bash
easygs train-colmap /path/to/scene \
  --save-dir /path/to/scene/3dgs \
  --alphas-folder /path/to/scene/skymask
```

带 depth / normal / extra attrs：

```bash
easygs train-colmap /path/to/scene \
  --save-dir /path/to/scene/3dgs \
  --depths-folder /path/to/scene/depths \
  --normals-folder /path/to/scene/normals \
  --extra-attrs-folder /path/to/scene/extra_attrs \
  --extra-attrs-dim 20
```

训练时不保存渲染图：

```bash
easygs train-colmap /path/to/scene \
  --save-dir /path/to/scene/3dgs \
  --no-save-rendered-images
```

查看完整参数：

```bash
easygs train-colmap --help
```

## 6. 训练输出在哪里

假设：

```bash
--save-dir /path/to/scene/3dgs
```

训练过程中会保存：

```text
/path/to/scene/3dgs/
  9999/
    model.ply
    cameras.pth
    extra_attrs.pth
    metadata.json
    rendered_images/
      ...
  19999/
    model.ply
    cameras.pth
    extra_attrs.pth
    metadata.json
```

注意：checkpoint 文件夹名是内部 iteration 下标。比如 `--save-interval 10000` 时，第一次保存通常是 `9999`，不是 `10000`。

`metadata.json` 包含：

- iteration
- active SH degree
- max SH degree
- Gaussian 点数量
- 最新 eval metrics
- 标量训练配置

## 7. 评估和渲染：CLI

评估某个 checkpoint：

```bash
easygs eval-colmap /path/to/scene /path/to/scene/3dgs/49999
```

同时保存渲染深度：

```bash
easygs eval-colmap /path/to/scene /path/to/scene/3dgs/49999 \
  --save-dir /path/to/scene/eval_49999 \
  --save-rendered-depth
```

只加载 `.ply`：

```bash
easygs eval-colmap /path/to/scene /path/to/model.ply \
  --save-dir /path/to/eval_model
```

评估输出目录里可能包含：

```text
0/
  rendered_images/
  rendered_depth/
  rendered_depth_wcolor/
  rendered_normals/
  rendered_alpha/
  rendered_extra_attrs/
```

## 8. 导出 COLMAP

如果你优化了相机，或者想把当前 cameras / points 写成 COLMAP text 格式：

```bash
easygs export-colmap /path/to/scene /path/to/scene/3dgs/49999 \
  --save-dir /path/to/exported_colmap
```

同时保存图像：

```bash
easygs export-colmap /path/to/scene /path/to/scene/3dgs/49999 \
  --save-dir /path/to/exported_colmap \
  --save-image
```

输出结构：

```text
/path/to/exported_colmap/
  sparse/
    0/
      cameras.txt
      images.txt
      points3D.txt
  images/          # 使用 --save-image 时才有
```

## 9. Python API：从 COLMAP 训练

最小示例：

```python
from EasyGS import GSer

trainer = GSer.from_colmap(
    colmap_path="/path/to/scene",
    save_dir="/path/to/scene/3dgs",
)
trainer.train()
```

推荐写法：使用配置对象。

```python
from EasyGS import GSer, LearningRateConfig, LossWeightsConfig, TrainConfig

train_cfg = TrainConfig(
    iterations=50_000,
    save_interval=10_000,
    eval_interval=10_000,
    eval_rate=0.1,
    enable_save_rendered_images=True,
)

lr_cfg = LearningRateConfig(
    position_lr_init=0.00016,
    scaling_lr=0.002,
)

trainer = GSer.from_colmap(
    colmap_path="/path/to/scene",
    images_folder="/path/to/scene/images",
    alphas_folder="/path/to/scene/skymask",
    save_dir="/path/to/scene/3dgs",
    height=520,
    width=930,
    lr_args=lr_cfg,
    loss_weights=LossWeightsConfig(),
    config=train_cfg,
)
trainer.train()
```

老式字典写法也兼容：

```python
from EasyGS import GSer, LearningRate, LossWeights

LearningRate["scaling_lr"] = 0.002

trainer = GSer(
    colmap_path="/path/to/scene",
    images_folder="/path/to/scene/images",
    save_dir="/path/to/scene/3dgs",
    lr_args=LearningRate,
    loss_weights=LossWeights,
)
trainer.train()
```

## 10. Python API：从数组训练

当相机来自 VGGT、DUSt3R、SLAM 或其他系统时，可以直接传数组：

```python
from EasyGS import GSer, TrainConfig

trainer = GSer.from_arrays(
    w2c=w2c.cpu().numpy(),               # [N, 4, 4] 或至少可取 [:3, :4]
    intrinsics=intrinsics.cpu().numpy(), # [N, 3, 3]
    images=images,                       # list[str] / list[PIL.Image] / np.ndarray
    xyz=points,                          # [P, 3]
    rgb=colors,                          # [P, 3], 可选，0-1 或 0-255
    save_dir="/path/to/output",
    config=TrainConfig(iterations=30_000),
)
trainer.train()
```

如果没有 images，只做可视化或评估，需要提供 `height` 和 `width`：

```python
trainer = GSer.from_arrays(
    w2c=w2c.cpu().numpy(),
    intrinsics=intrinsics.cpu().numpy(),
    images=None,
    xyz=points,
    height=720,
    width=1280,
    save_dir="/path/to/output",
)
try:
    metrics = trainer.eval()
finally:
    trainer.close()
```

## 11. 加载预训练模型

`pretrained_path` 支持两种形式。

单个 PLY：

```python
trainer = GSer.from_colmap(
    colmap_path="/path/to/scene",
    pretrained_path="/path/to/model.ply",
    save_dir="/path/to/output",
)
```

checkpoint 目录：

```python
trainer = GSer.from_colmap(
    colmap_path="/path/to/scene",
    pretrained_path="/path/to/scene/3dgs/49999",
    save_dir="/path/to/output",
)
```

checkpoint 目录中如果存在这些文件，会自动加载：

```text
model.ply
cameras.pth
extra_attrs.pth
```

如果只有 `model.ply`，也可以加载；相机会使用当前场景的 COLMAP 相机。

## 12. 相机优化

场景 Gaussian 固定，只优化相机参数：

```python
from EasyGS import GSer, TrainConfig

cfg = TrainConfig(
    enable_freeze=True,
    enable_cam_update=True,
    use_sift_loss=True,
    cam_update_from_iter=0,
    iterations=10_000,
    save_interval=5_000,
)

trainer = GSer.from_colmap(
    colmap_path="/path/to/scene",
    pretrained_path="/path/to/model_or_checkpoint",
    save_dir="/path/to/camera_optimized",
    config=cfg,
)
trainer.train()
trainer.save_colmap(trainer.save_dir, save_image=True)
```

命令行版本：

```bash
easygs train-colmap /path/to/scene \
  --pretrained-path /path/to/model_or_checkpoint \
  --save-dir /path/to/camera_optimized \
  --freeze \
  --cam-update \
  --cam-update-from-iter 0 \
  --sift-loss \
  --iterations 10000
```

注意：当前 SIFT reprojection loss 主要作为辅助项，内部使用 OpenCV 特征匹配，并不是完整的端到端可微几何 BA。高精度相机优化建议后续接入可微重投影或 SE(3) 增量优化。

## 13. 只拟合 SH 颜色

固定几何，只调整颜色/SH：

```python
from EasyGS import GSer, TrainConfig

cfg = TrainConfig(
    only_fit_sh=True,
    enable_densification=False,
    enable_reset_opacity=False,
    iterations=10_000,
)

trainer = GSer.from_colmap(
    colmap_path="/path/to/scene",
    pretrained_path="/path/to/model_or_checkpoint",
    save_dir="/path/to/fit_sh",
    config=cfg,
)
trainer.train()
```

命令行：

```bash
easygs train-colmap /path/to/scene \
  --pretrained-path /path/to/model_or_checkpoint \
  --save-dir /path/to/fit_sh \
  --only-fit-sh \
  --no-densification \
  --no-reset-opacity
```

## 14. 常用配置解释

训练长度和保存：

```python
TrainConfig(
    iterations=50_000,
    save_interval=10_000,
    eval_interval=10_000,
)
```

训练/验证划分：

```python
TrainConfig(
    eval_rate=0.1,          # 每 10 张取 1 张做 eval
    enable_train_all=True,  # eval 图像也参与训练
)
```

densification：

```python
TrainConfig(
    enable_densification=True,
    densify_from_iter=500,
    densify_until_iter=15_000,
    densification_interval=100,
    densify_grad_threshold=0.0002,
    n_split=2,
)
```

opacity reset：

```python
TrainConfig(
    enable_reset_opacity=True,
    opacity_reset_interval=3_000,
    opacity_reset_until_iter=15_000,
)
```

输出控制：

```python
TrainConfig(
    enable_save_rendered_images=True,
    enable_save_rendered_depth=False,
    enable_save_rendered_normals=False,
    enable_save_rendered_alpha=False,
    enable_save_rendered_extra_attrs=False,
)
```

学习率：

```python
LearningRateConfig(
    position_lr_init=0.00016,
    position_lr_final=0.0000016,
    scaling_lr=0.005,
    feature_lr=0.0025,
    opacity_lr=0.025,
    rotation_lr=0.001,
)
```

loss 权重：

```python
LossWeightsConfig(
    rgb_l1_weight=0.8,
    rgb_ssim_weight=0.2,
    depth_weight=1.0,
    normals_weight=1.0,
    alpha_weight=1.0,
    sift_weight=1.0,
    extra_attrs_weight=0.1,
)
```

## 15. Callbacks：自定义训练行为

默认 callbacks 包括：

- progress bar
- 定期 eval
- 定期 checkpoint
- eval 结果打印
- wandb logging，可选

如果你传 `callbacks=[...]`，会替换默认 callbacks：

```python
from EasyGS import Callback, GSer


class LossPrinter(Callback):
    def on_train_iteration_end(self, trainer, iteration, loss_scalars, render_pkg):
        if iteration % 100 == 0:
            print(iteration, loss_scalars["total"])


trainer = GSer.from_colmap(
    colmap_path="/path/to/scene",
    save_dir="/path/to/output",
    callbacks=[LossPrinter()],
    enable_progress=False,
)
trainer.train()
```

如果想保留默认 callbacks，再额外加自己的 callback，用 `extra_callbacks`：

```python
trainer = GSer.from_colmap(
    colmap_path="/path/to/scene",
    save_dir="/path/to/output",
    extra_callbacks=[LossPrinter()],
)
trainer.train()
```

常用 hook：

```python
class MyCallback(Callback):
    def setup(self, trainer):
        pass

    def on_train_start(self, trainer):
        pass

    def on_train_iteration_end(self, trainer, iteration, loss_scalars, render_pkg):
        pass

    def on_eval_end(self, trainer, iteration, metrics):
        pass

    def on_train_end(self, trainer):
        pass
```

## 16. wandb 记录

安装：

```bash
pip install -e ".[logging]"
```

Python：

```python
trainer = GSer.from_colmap(
    colmap_path="/path/to/scene",
    save_dir="/path/to/output",
    wandb_project="EasyGS",
    wandb_name="scene_001",
)
trainer.train()
```

CLI：

```bash
easygs train-colmap /path/to/scene \
  --save-dir /path/to/output \
  --wandb-project EasyGS \
  --wandb-name scene_001
```

## 17. 二次开发建议

加一个 robust depth loss：

1. 打开 `losses.py`
2. 在 `GaussianLossComputer.__call__` 中替换 depth loss
3. 仍然返回 `total, scalars`

加一种新的数据来源：

1. 打开 `data.py`
2. 增加新的 `_build_from_xxx`
3. 返回 `SceneBuildResult(gaussians=..., cameras=...)`

加一种新的 checkpoint 内容：

1. 打开 `checkpoint.py`
2. 修改 `CheckpointManager.save`
3. 同步写进 `metadata.json`

加训练过程中的新行为：

1. 新建一个 `Callback`
2. 实现对应 hook
3. 通过 `extra_callbacks=[...]` 传给 `GSer`

## 18. 常见问题

### 1. `diff_gauss_pose` 找不到

说明 rasterizer 扩展没有安装成功：

```bash
pip install submodules/diff-gaussian-rasterization
```

### 2. `simple_knn._C` 找不到

说明 KNN CUDA 扩展没有安装成功：

```bash
pip install submodules/simple-knn
```

### 3. COLMAP cameras 找不到

确认存在：

```text
sparse/0/images.bin
sparse/0/cameras.bin
```

或：

```text
sparse/0/images.txt
sparse/0/cameras.txt
```

### 4. 没有 `points3D.bin/txt`

如果从零训练，需要：

```text
sparse/0/points3D.bin
```

或：

```text
sparse/0/points3D.txt
```

如果没有点云，请提供：

```python
pretrained_path="/path/to/model.ply"
```

或使用 `from_arrays(..., xyz=points)`。

### 5. checkpoint 文件夹名为什么是 `9999`

训练循环内部 iteration 从 0 开始。`--save-interval 10000` 时，第一次保存发生在第 10000 步结束，对应内部下标 `9999`。

### 6. 如何关闭进度条

CLI：

```bash
easygs train-colmap /path/to/scene --no-progress
```

Python：

```python
trainer = GSer.from_colmap(
    colmap_path="/path/to/scene",
    enable_progress=False,
)
```

### 7. 如何只评估不训练

```bash
easygs eval-colmap /path/to/scene /path/to/checkpoint
```

或：

```python
trainer = GSer.from_colmap(
    colmap_path="/path/to/scene",
    pretrained_path="/path/to/checkpoint",
    save_dir="/path/to/eval",
)
try:
    metrics = trainer.eval()
finally:
    trainer.close()
```

### 8. 可以 CPU 跑吗

目前不建议。核心 rasterizer 和 KNN 扩展依赖 CUDA，代码中底层模型也默认使用 CUDA。

## 19. 兼容说明

- `GSer(...)` 原始入口仍然可用。
- `LearningRate` / `LossWeights` 字典仍然导出，旧脚本可以继续改字典。
- `srcipts/` 目录保留用于兼容原始示例；新代码建议使用 `examples/` 或 CLI。
- `eval()` 返回 `{"psnr": ..., "ssim": ..., "l1": ...}`。
- `save_outputs(...)` 仍保留，但内部已经委托给 `RenderOutputSaver`。
- `save_colmap(...)` 仍保留，但内部已经委托给 `save_colmap_reconstruction(...)`。
