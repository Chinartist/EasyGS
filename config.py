from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class LearningRateConfig:
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    scaling_lr: float = 0.005
    qvec_lr: float = 0.001
    tvec_lr: float = 0.05
    cam_lr: float = 0.005
    extra_attrs_lr: float = 0.001
    feature_lr: float = 0.0025
    opacity_lr: float = 0.025
    rotation_lr: float = 0.001

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class LossWeightsConfig:
    rgb_l1_weight: float = 0.8
    rgb_ssim_weight: float = 0.2
    depth_weight: float = 1.0
    normals_weight: float = 1.0
    alpha_weight: float = 1.0
    sift_weight: float = 1.0
    extra_attrs_weight: float = 0.1

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class TrainConfig:
    preload: bool = True
    enable_freeze: bool = False
    only_fit_sh: bool = False
    add_skybox: bool = False
    enable_densification: bool = True
    enable_reset_opacity: bool = True
    enable_train_all: bool = True
    enable_cam_update: bool = False
    enable_save_skybox: bool = False
    enable_save_rendered_images: bool = True
    enable_save_rendered_depth: bool = False
    enable_save_rendered_normals: bool = False
    enable_save_rendered_alpha: bool = False
    enable_save_rendered_extra_attrs: bool = False
    verbose: bool = True
    sem_ignore_index: int = 255
    n_split: int = 2
    min_opacity: float = 0.005
    percent_dense: float = 0.01
    skybox_points: int = 100_000
    skybox_radius_scale: float = 10.0
    extra_attrs_dim: int = 0
    eval_rate: float = 1.0
    use_sift_loss: bool = False
    init_degree: int = 0
    max_sh_degree: int = 3
    bg_color: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    iterations: int = 30_000
    sh_increase_interval: int = 1000
    save_interval: int = 10_000
    eval_interval: int = 1000
    opacity_reset_interval: int = 3000
    cam_update_from_iter: int = 10_000
    densify_from_iter: int = 500
    densification_interval: int = 100
    densify_until_iter: int = 15_000
    opacity_reset_until_iter: int = 15_000
    densify_grad_threshold: float = 0.0002
    wandb_project: Optional[str] = None
    wandb_name: Optional[str] = None

    def to_kwargs(self) -> Dict[str, object]:
        kwargs = asdict(self)
        kwargs["N_split"] = kwargs.pop("n_split")
        return kwargs
