import os
import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from EasyGS import GSer, LearningRate, LossWeights
LearningRate["scaling_lr_init"] = 0.002
LearningRate["position_lr_init"] = 0.00016

data_dir="/home/tangyuan/wks/aggressive/calib_project/EasyGS/data/Instance_1_R39_1752221033300" 
trainer = GSer(
        #要么提供COLMAP路径，要么提供相机参数
        colmap_path=f"{data_dir}",
        images_folder=f"{data_dir}/images",
        alphas_folder=f"{data_dir}/masks",
        pretrained_path="/home/tangyuan/wks/aggressive/calib_project/EasyGS/data/Instance_1_R39_1752221033300/Instance_0.ply",
        save_dir=f"{data_dir}/fit_sh",
        #训练和测试设置
        preload=True,
        enable_freeze=False,
        only_fit_sh = True,
        enable_densification=False,
        enable_reset_opacity=False,
        enable_train_all=True,
        enable_cam_update = True,

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
        eval_rate=1,
        lr_args=LearningRate,
        loss_weights=LossWeights,
        init_degree=0,
        max_sh_degree=0,
        bg_color = [1, 1, 1],
        sh_increase_interval = 1000,
        densify_from_iter=0,
        densify_until_iter=45_000,
        opacity_reset_until_iter=40_000,
        iterations=2000,
        save_interval=2_000,
        eval_interval=500,
        densification_interval=100,
        opacity_reset_interval=3000,
        cam_update_from_iter = 0,
        wandb_project="PPGS")
trainer.train()