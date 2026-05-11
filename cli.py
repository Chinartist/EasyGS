from __future__ import annotations

import argparse
import os

if __package__:
    from .Trainer import GSer
    from .config import LearningRateConfig, LossWeightsConfig, TrainConfig
else:
    from Trainer import GSer
    from config import LearningRateConfig, LossWeightsConfig, TrainConfig


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


def build_parser():
    parser = argparse.ArgumentParser(prog="easygs", description="EasyGS training and utility commands.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train-colmap", help="Train a Gaussian model from a COLMAP scene.")
    train.add_argument("scene", help="COLMAP scene root containing sparse/0 and images.")
    _add_scene_args(train)
    _add_train_args(train)
    train.set_defaults(func=train_colmap)

    eval_parser = subparsers.add_parser("eval-colmap", help="Evaluate/render a pretrained model on a COLMAP scene.")
    eval_parser.add_argument("scene", help="COLMAP scene root containing sparse/0 and images.")
    eval_parser.add_argument("checkpoint", help="Checkpoint directory or .ply file.")
    _add_scene_args(eval_parser)
    _add_eval_args(eval_parser)
    eval_parser.set_defaults(func=eval_colmap)

    export = subparsers.add_parser("export-colmap", help="Export optimized cameras and points to COLMAP text format.")
    export.add_argument("scene", help="COLMAP scene root used to build cameras.")
    export.add_argument("checkpoint", help="Checkpoint directory or .ply file.")
    _add_scene_args(export)
    export.add_argument("--save-dir", required=True, help="Output COLMAP directory.")
    export.add_argument("--save-image", action="store_true", help="Also write images into the output directory.")
    export.add_argument("--no-progress", action="store_true", help="Disable progress bars.")
    export.set_defaults(func=export_colmap)

    return parser


def train_colmap(args):
    config = TrainConfig(
        preload=not args.no_preload,
        enable_freeze=args.freeze,
        only_fit_sh=args.only_fit_sh,
        add_skybox=args.add_skybox,
        enable_densification=not args.no_densification,
        enable_reset_opacity=not args.no_reset_opacity,
        enable_train_all=not args.train_split_only,
        enable_cam_update=args.cam_update,
        enable_save_skybox=args.save_skybox,
        enable_save_rendered_images=not args.no_save_rendered_images,
        enable_save_rendered_depth=args.save_rendered_depth,
        enable_save_rendered_normals=args.save_rendered_normals,
        enable_save_rendered_alpha=args.save_rendered_alpha,
        enable_save_rendered_extra_attrs=args.save_rendered_extra_attrs,
        verbose=not args.quiet,
        sem_ignore_index=args.sem_ignore_index,
        n_split=args.n_split,
        min_opacity=args.min_opacity,
        percent_dense=args.percent_dense,
        skybox_points=args.skybox_points,
        skybox_radius_scale=args.skybox_radius_scale,
        extra_attrs_dim=args.extra_attrs_dim,
        eval_rate=args.eval_rate,
        use_sift_loss=args.sift_loss,
        init_degree=args.init_degree,
        max_sh_degree=args.max_sh_degree,
        bg_color=args.bg_color,
        iterations=args.iterations,
        sh_increase_interval=args.sh_increase_interval,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        opacity_reset_interval=args.opacity_reset_interval,
        cam_update_from_iter=args.cam_update_from_iter,
        densify_from_iter=args.densify_from_iter,
        densification_interval=args.densification_interval,
        densify_until_iter=args.densify_until_iter,
        opacity_reset_until_iter=args.opacity_reset_until_iter,
        densify_grad_threshold=args.densify_grad_threshold,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
    )
    trainer = GSer.from_colmap(
        colmap_path=args.scene,
        save_dir=args.save_dir,
        images_folder=args.images_folder,
        depths_folder=args.depths_folder,
        normals_folder=args.normals_folder,
        alphas_folder=args.alphas_folder,
        extra_attrs_folder=args.extra_attrs_folder,
        pretrained_path=args.pretrained_path,
        height=args.height,
        width=args.width,
        lr_args=_learning_rate_config(args),
        loss_weights=LossWeightsConfig(),
        config=config,
        enable_progress=not args.no_progress,
    )
    trainer.train()


def eval_colmap(args):
    save_dir = args.save_dir or _default_eval_dir(args.checkpoint)
    config = TrainConfig(
        iterations=1,
        eval_rate=1.0,
        enable_train_all=True,
        enable_densification=False,
        enable_reset_opacity=False,
        enable_save_rendered_images=not args.no_save_rendered_images,
        enable_save_rendered_depth=args.save_rendered_depth,
        enable_save_rendered_normals=args.save_rendered_normals,
        enable_save_rendered_alpha=args.save_rendered_alpha,
        enable_save_rendered_extra_attrs=args.save_rendered_extra_attrs,
    )
    trainer = GSer.from_colmap(
        colmap_path=args.scene,
        pretrained_path=args.checkpoint,
        save_dir=save_dir,
        images_folder=args.images_folder,
        depths_folder=args.depths_folder,
        normals_folder=args.normals_folder,
        alphas_folder=args.alphas_folder,
        extra_attrs_folder=args.extra_attrs_folder,
        height=args.height,
        width=args.width,
        config=config,
        enable_progress=not args.no_progress,
    )
    try:
        trainer.eval(args.iteration)
    finally:
        trainer.close()


def export_colmap(args):
    save_dir = args.save_dir
    config = TrainConfig(
        iterations=1,
        eval_rate=1.0,
        enable_train_all=True,
        enable_densification=False,
        enable_reset_opacity=False,
        enable_save_rendered_images=False,
    )
    trainer = GSer.from_colmap(
        colmap_path=args.scene,
        pretrained_path=args.checkpoint,
        save_dir=save_dir,
        images_folder=args.images_folder,
        depths_folder=args.depths_folder,
        normals_folder=args.normals_folder,
        alphas_folder=args.alphas_folder,
        extra_attrs_folder=args.extra_attrs_folder,
        height=args.height,
        width=args.width,
        config=config,
        enable_progress=not args.no_progress,
    )
    try:
        trainer.save_colmap(save_dir, save_image=args.save_image)
    finally:
        trainer.close()


def _add_scene_args(parser):
    parser.add_argument("--images-folder", default=None)
    parser.add_argument("--depths-folder", default=None)
    parser.add_argument("--normals-folder", default=None)
    parser.add_argument("--alphas-folder", default=None)
    parser.add_argument("--extra-attrs-folder", default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)


def _add_train_args(parser):
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--pretrained-path", default=None)
    parser.add_argument("--iterations", type=int, default=30_000)
    parser.add_argument("--save-interval", type=int, default=10_000)
    parser.add_argument("--eval-interval", type=int, default=1_000)
    parser.add_argument("--eval-rate", type=float, default=1.0)
    parser.add_argument("--sh-increase-interval", type=int, default=1_000)
    parser.add_argument("--opacity-reset-interval", type=int, default=3_000)
    parser.add_argument("--opacity-reset-until-iter", type=int, default=15_000)
    parser.add_argument("--densify-from-iter", type=int, default=500)
    parser.add_argument("--densify-until-iter", type=int, default=15_000)
    parser.add_argument("--densification-interval", type=int, default=100)
    parser.add_argument("--densify-grad-threshold", type=float, default=0.0002)
    parser.add_argument("--cam-update-from-iter", type=int, default=10_000)
    parser.add_argument("--percent-dense", type=float, default=0.01)
    parser.add_argument("--min-opacity", type=float, default=0.005)
    parser.add_argument("--n-split", type=int, default=2)
    parser.add_argument("--init-degree", type=int, default=0)
    parser.add_argument("--max-sh-degree", type=int, default=3)
    parser.add_argument("--extra-attrs-dim", type=int, default=0)
    parser.add_argument("--sem-ignore-index", type=int, default=255)
    parser.add_argument("--skybox-points", type=int, default=100_000)
    parser.add_argument("--skybox-radius-scale", type=float, default=10.0)
    parser.add_argument("--bg-color", nargs=3, type=float, default=[1.0, 1.0, 1.0])
    lr_defaults = LearningRateConfig()
    parser.add_argument("--position-lr-init", type=float, default=lr_defaults.position_lr_init)
    parser.add_argument("--scaling-lr", type=float, default=lr_defaults.scaling_lr)
    parser.add_argument("--feature-lr", type=float, default=lr_defaults.feature_lr)
    parser.add_argument("--opacity-lr", type=float, default=lr_defaults.opacity_lr)
    parser.add_argument("--rotation-lr", type=float, default=lr_defaults.rotation_lr)
    parser.add_argument("--qvec-lr", type=float, default=lr_defaults.qvec_lr)
    parser.add_argument("--tvec-lr", type=float, default=lr_defaults.tvec_lr)
    parser.add_argument("--no-preload", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--only-fit-sh", action="store_true")
    parser.add_argument("--add-skybox", action="store_true")
    parser.add_argument("--no-densification", action="store_true")
    parser.add_argument("--no-reset-opacity", action="store_true")
    parser.add_argument("--train-split-only", action="store_true", help="Do not train on eval cameras.")
    parser.add_argument("--cam-update", action="store_true")
    parser.add_argument("--sift-loss", action="store_true")
    parser.add_argument("--save-skybox", action="store_true")
    parser.add_argument("--no-save-rendered-images", action="store_true")
    parser.add_argument("--save-rendered-depth", action="store_true")
    parser.add_argument("--save-rendered-normals", action="store_true")
    parser.add_argument("--save-rendered-alpha", action="store_true")
    parser.add_argument("--save-rendered-extra-attrs", action="store_true")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--quiet", action="store_true")


def _add_eval_args(parser):
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--no-save-rendered-images", action="store_true")
    parser.add_argument("--save-rendered-depth", action="store_true")
    parser.add_argument("--save-rendered-normals", action="store_true")
    parser.add_argument("--save-rendered-alpha", action="store_true")
    parser.add_argument("--save-rendered-extra-attrs", action="store_true")
    parser.add_argument("--no-progress", action="store_true")


def _learning_rate_config(args):
    defaults = LearningRateConfig()
    return LearningRateConfig(
        position_lr_init=args.position_lr_init,
        scaling_lr=args.scaling_lr,
        feature_lr=args.feature_lr,
        opacity_lr=args.opacity_lr,
        rotation_lr=args.rotation_lr,
        qvec_lr=args.qvec_lr,
        tvec_lr=args.tvec_lr,
        position_lr_final=defaults.position_lr_final,
        position_lr_delay_mult=defaults.position_lr_delay_mult,
        cam_lr=defaults.cam_lr,
        extra_attrs_lr=defaults.extra_attrs_lr,
    )


def _default_eval_dir(checkpoint):
    if checkpoint.endswith(".ply"):
        root, _ = os.path.splitext(checkpoint)
        return f"{root}_eval"
    return os.path.join(checkpoint, "eval")


if __name__ == "__main__":
    main()
