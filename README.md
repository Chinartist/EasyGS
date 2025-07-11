# Plug-and-Play 3DGS
## Install
```bash
pip install pycolmap==3.11
pip install wandb
```
## How to train with COLMAP format data
``` python
from EasyGS import GSer, LearningRate, LossWeights
LearningRate["scaling_lr_init"] = 0.002
LearningRate["position_lr_init"] = 0.00016
trainer = GSer(
        #要么提供COLMAP路径，要么提供相机参数
        colmap_path=f"{data_dir}",
        images_folder=f"{data_dir}/images",
        alphas_folder=f"{data_dir}/skymask",
        # pretrained_path="/home/tangyuan/project/data/static_scene/merged.ply",#pretrained model path, if you want to use a pretrained model, set this. pretrained_path can also be ply file
        save_dir=f"{data_dir}/3dgs",
        height=520,#if you want to resize the image, you need to set this
        width=930,
        #训练和测试设置
        preload=True,
        enable_densification=True,
        enable_reset_opacity=True,
        enable_train_all=True,
        enable_save_rendered_images=True,
        enable_save_rendered_depth=False,
        enable_save_rendered_normals=False,
        enable_save_rendered_alpha=False,
        enable_save_rendered_extra_attrs=False,
        verbose=True,
        #训练参数
        N_split = 4,
        percent_dense = 0.0005,
        extra_attrs_dim=0,
        eval_rate=0.1,
        lr_args=LearningRate,
        loss_weights=LossWeights,
       
        init_degree=0,
        max_sh_degree=3,
        bg_color = [1, 1, 1],
        sh_increase_interval = 1000,
        densify_from_iter=0,
        densify_until_iter=45_000,
        opacity_reset_until_iter=40_000,
        iterations=50_000,
        save_interval=10_000,
        eval_interval=10_000,
        
        densification_interval=100,
        opacity_reset_interval=3000,
        
        wandb_project=None)
trainer.train()
```
## How to train with custom data
```python
from EasyGS import GSer, LearningRate, LossWeights
#w2c and intrinsics maybe come from vggt, dust3r or other methods
trainer = GSer(
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

from EasyGS import GSer
trainer = GSer(
    w2c=w2c.cpu().numpy(),
    intrinsics=intrinsics_pinhole.cpu().numpy(),
    height=height,#if you don't provide images, you need to set height and width corrosponding to the intrinsics
    width=width,
    xyz=xyz, #if you don't provide a pretrained model dir or a ply file, you need to provide xyz
    save_dir= out_dir) 
trainer.eval()
```