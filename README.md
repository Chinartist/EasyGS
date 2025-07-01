# Plug-and-Play 3DGS
## How to train with COLMAP format data
``` python
from PPGS.Trainer import GSTrainer, LearningRate, LossWeights
trainer = GSTrainer(
        #要么提供COLMAP路径，要么提供相机参数
        colmap_path="./inputs/full/figurines",
        pretrained_path="./outputs/figurines/900",#pretrained model path, if you want to use a pretrained model, set this. pretrained_path can also be ply file
        save_dir="./outputs/figurines",
        height=450,#if you want to resize the image, you need to set this
        width=800,
        #训练和测试设置
        enable_densification=True,
        enable_reset_opacity=True,
        enable_train_all=True,
        enable_cam_update=False,#if you want to update the camera parameters, set this to True
        enable_save_rendered_images=True,
        enable_save_rendered_depth=True,
        enable_save_rendered_normals=False,
        enable_save_rendered_alpha=False,
        enable_save_rendered_extra_attrs=False,

        #训练参数
        extra_attrs_dim=0,
        eval_rate=0.1,
        lr_args=LearningRate,
        loss_weights=LossWeights,
       
        init_degree=0,
        max_sh_degree=3,
        bg_color = [1, 1, 1],
        iterations=10_000,
        save_interval=1000,
        eval_interval=250)
trainer.train()
```
## How to train with custom data
```python
from PPGS.Trainer import GSTrainer, LearningRate, LossWeights
#w2c and intrinsics maybe come from vggt, dust3r or other methods
trainer = GSTrainer(
    w2c=w2c.cpu().numpy(),
    intrinsics=intrinsics_pinhole.cpu().numpy(),
    images=images,
    xyz=points, 
    save_dir= out_dir) 
trainer.train()
# If you want to save the COLMAP format data
trainer.save_colmap(trainer.save_dir, save_image=True)
```
## How to visualize only
```python

from PPGS.Trainer import GSTrainer
trainer = GSTrainer(
    w2c=w2c.cpu().numpy(),
    intrinsics=intrinsics_pinhole.cpu().numpy(),
    height=height,#if you don't provide images, you need to set height and width corrosponding to the intrinsics
    width=width,
    xyz=xyz, #if you don't provide a pretrained model dir or a ply file, you need to provide xyz
    save_dir= out_dir) 
trainer.eval()
```