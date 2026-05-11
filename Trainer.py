import os
import torch
import random
from typing import Mapping

if __package__:
    from .gaussian_renderer import render
    from .callbacks import CallbackList, default_callbacks
    from .colmap_io import save_colmap_reconstruction
    from .config import LearningRateConfig, LossWeightsConfig
    from .checkpoint import CheckpointManager, load_pretrained_checkpoint
    from .data import build_scene
    from .losses import EvalMetricComputer, GaussianLossComputer
    from .outputs import RenderOutputSaver
else:
    from gaussian_renderer import render
    from callbacks import CallbackList, default_callbacks
    from colmap_io import save_colmap_reconstruction
    from config import LearningRateConfig, LossWeightsConfig
    from checkpoint import CheckpointManager, load_pretrained_checkpoint
    from data import build_scene
    from losses import EvalMetricComputer, GaussianLossComputer
    from outputs import RenderOutputSaver
from copy import deepcopy
from torch.optim.lr_scheduler import LinearLR

LearningRate = LearningRateConfig().to_dict()
LossWeights = LossWeightsConfig().to_dict()


class GSer:
    @staticmethod
    def _normalize_config(config, defaults):
        if config is None:
            return dict(defaults)
        if hasattr(config, "to_dict"):
            return config.to_dict()
        if isinstance(config, Mapping):
            merged = dict(defaults)
            merged.update(config)
            return merged
        raise TypeError("Config must be a dict-like object or expose to_dict().")

    def _collect_run_config(self):
        run_config = {}
        for key, value in self.__dict__.items():
            if isinstance(value, (str, int, float, bool)):
                run_config[key] = value
            elif isinstance(value, (list, tuple)) and all(isinstance(item, (str, int, float, bool)) for item in value):
                run_config[key] = list(value)
        return run_config

    @classmethod
    def from_colmap(cls, colmap_path, save_dir=None, config=None, **kwargs):
        """Create a trainer from a COLMAP scene with readable defaults."""
        if save_dir is None:
            save_dir = os.path.join(colmap_path, "3dgs")
        if config is not None:
            if not hasattr(config, "to_kwargs"):
                raise TypeError("config must expose to_kwargs().")
            kwargs = {**config.to_kwargs(), **kwargs}
        return cls(colmap_path=colmap_path, save_dir=save_dir, **kwargs)

    @classmethod
    def from_arrays(cls, w2c, intrinsics, images=None, xyz=None, save_dir="outputs", config=None, **kwargs):
        """Create a trainer from in-memory cameras, images, and points."""
        if config is not None:
            if not hasattr(config, "to_kwargs"):
                raise TypeError("config must expose to_kwargs().")
            kwargs = {**config.to_kwargs(), **kwargs}
        return cls(
            w2c=w2c,
            intrinsics=intrinsics,
            images=images,
            xyz=xyz,
            save_dir=save_dir,
            **kwargs,
        )

    def __init__(self,
                 # data sources
                 colmap_path=None,
                 w2c=None,
                 intrinsics=None,
                 images=None,
                 xyz=None,
                 rgb=None,
                 pretrained_path=None,
                 save_dir=None,
                 height=None,
                 width=None,
                 # optional data folders
                 images_folder=None,
                 depths_folder=None,
                 normals_folder=None,
                 alphas_folder=None,
                 extra_attrs_folder=None,
                 # data loading and training options
                 preload=True,
                 enable_freeze=False,
                 only_fit_sh=False,
                 add_skybox=False,
                 enable_densification=True,
                 enable_reset_opacity=True,
                 enable_train_all=True,
                 enable_cam_update=False,
                 enable_save_skybox=False,
                 enable_save_rendered_images=True,
                 enable_save_rendered_depth=False,
                 enable_save_rendered_normals=False,
                 enable_save_rendered_alpha=False,
                 enable_save_rendered_extra_attrs=False,

                 # training parameters
                 verbose=True,
                 sem_ignore_index=255,
                 N_split=2,
                 min_opacity=0.005,
                 percent_dense=0.01,
                 skybox_points=100_000, skybox_radius_scale=10.0,
                 extra_attrs_dim=0,
                 eval_rate=1.0,
                 lr_args=None,
                 loss_weights=None,

                 use_sift_loss=False,

                 init_degree=0,
                 max_sh_degree=3,
                 bg_color=None,

                 iterations=30_000,
                 sh_increase_interval=1000,

                 save_interval=10_000,
                 eval_interval=1000,
                 opacity_reset_interval=3000,
                 cam_update_from_iter=10000,
                 densify_from_iter=500,
                 densification_interval=100,
                 densify_until_iter=15_000,
                 opacity_reset_until_iter=15_000,
                 densify_grad_threshold=0.0002,
                 wandb_project=None,
                 wandb_name=None,
                 callbacks=None,
                 extra_callbacks=None,
                 enable_progress=True,

                 ):
        if not 0 < eval_rate <= 1:
            raise ValueError("eval_rate must be in the range (0, 1].")
        if iterations <= 0:
            raise ValueError("iterations must be greater than 0.")
        for name, value in {
            "sh_increase_interval": sh_increase_interval,
            "save_interval": save_interval,
            "eval_interval": eval_interval,
            "opacity_reset_interval": opacity_reset_interval,
            "densification_interval": densification_interval,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than 0.")
        if enable_cam_update and iterations <= cam_update_from_iter:
            raise ValueError(
                "enable_cam_update=True but iterations <= cam_update_from_iter, so the camera optimizer "
                "would never step. Lower cam_update_from_iter or increase iterations."
            )

        if enable_freeze or only_fit_sh:
            if enable_densification:
                print("Densification is disabled when enable_freeze=True or only_fit_sh=True.")
                enable_densification = False
            if enable_reset_opacity:
                print("Opacity reset is disabled when enable_freeze=True or only_fit_sh=True.")
                enable_reset_opacity = False

        lr_args = self._normalize_config(lr_args, LearningRate)
        loss_weights = self._normalize_config(loss_weights, LossWeights)
        bg_color = [1, 1, 1] if bg_color is None else list(bg_color)
        if save_dir is None:
            save_dir = "outputs"
        os.makedirs(save_dir, exist_ok=True)
        self.device = torch.device("cuda")
        lr_args["position_lr_max_steps"] = iterations
        print(lr_args)
        print(loss_weights)

        scene = build_scene(
            colmap_path=colmap_path,
            w2c=w2c,
            intrinsics=intrinsics,
            images=images,
            xyz=xyz,
            rgb=rgb,
            pretrained_path=pretrained_path,
            height=height,
            width=width,
            images_folder=images_folder,
            depths_folder=depths_folder,
            normals_folder=normals_folder,
            alphas_folder=alphas_folder,
            extra_attrs_folder=extra_attrs_folder,
            preload=preload,
            init_degree=init_degree,
            max_sh_degree=max_sh_degree,
            extra_attrs_dim=extra_attrs_dim,
            percent_dense=percent_dense,
            verbose=verbose,
            add_skybox=add_skybox,
            skybox_points=skybox_points,
            skybox_radius_scale=skybox_radius_scale,
        )
        gaussians = scene.gaussians
        cams = scene.cameras

        # load pretrained Gaussian model
        if pretrained_path is not None:
            load_pretrained_checkpoint(pretrained_path, gaussians, cams, self.device)

        # split cameras for training and evaluation
        if eval_rate < 1:
            self.indices_for_eval = [idx for idx in range(len(cams)) if idx % int(1 / eval_rate) == 0]
        else:
            print("eval_rate is set to 1, all cameras will be used for evaluation.")
            self.indices_for_eval = [idx for idx in range(len(cams))]
        if not enable_train_all:
            self.indices_for_train = [idx for idx in range(len(cams)) if idx not in self.indices_for_eval]
        else:
            print("enable_train_all is set to True, all cameras will be used for training.")
            self.indices_for_train = [idx for idx in range(len(cams))]
        if not self.indices_for_train:
            raise ValueError("No cameras selected for training. Set enable_train_all=True or lower eval_rate.")
        if not self.indices_for_eval:
            raise ValueError("No cameras selected for evaluation. Increase eval_rate.")
        print(f"Number of cameras: {len(cams)}")
        print(
            f"Number of cameras for training: {len(self.indices_for_train)}, for evaluation: {len(self.indices_for_eval)}")

        # configure camera optimization
        self.enable_cam_update = enable_cam_update
        self.use_sift_loss = use_sift_loss
        if not enable_cam_update:
            cams.requires_grad_(False)
            print("enable_cam_update is set to False, cameras will not be updated during training.")
            self.optimizer_cam = None
        else:
            print("enable_cam_update is set to True, cameras will be updated during training.")
            # self.optimizer_cam = torch.optim.Adam(cams.parameters(), lr=lr_args["cam_lr"])
            qvec_params = []
            tvec_params = []
            for cam in cams:
                qvec_params.append(cam.qvec)
                tvec_params.append(cam.tvec)
            self.optimizer_cam = torch.optim.Adam([
                {'params': qvec_params, 'lr': lr_args["qvec_lr"]},
                {'params': tvec_params, 'lr': lr_args["tvec_lr"]},
            ])

            self.scheduler_cam = LinearLR(self.optimizer_cam, start_factor=1.0, end_factor=0.01,
                                          total_iters=max(1, iterations - cam_update_from_iter))
        # print selected training settings
        if not enable_densification:
            print("enable_densification is set to False, densification will not be performed during training.")
        if not enable_reset_opacity:
            print("enable_reset_opacity is set to False, opacity will not be reset during training.")
        print(f"Model will be saved to {save_dir}")
        print(f"Scene extent: {gaussians.scene_extent}")
        # configure Gaussian optimizer
        gaussians.training_setup(lr_args, freeze=enable_freeze, only_fit_sh=only_fit_sh)
        if gaussians.skyboxer is not None:
            lr_skyboxer = {k: v for k, v in lr_args.items()}
            lr_skyboxer["extra_attrs_lr"] = 0.
            gaussians.skyboxer.training_setup(lr_skyboxer)

        # initialize instance state
        self.gaussians = gaussians
        self.cams = cams
        self.N_split = N_split
        self.min_opacity = min_opacity
        self.enable_densification = enable_densification
        self.enable_freeze = enable_freeze
        self.enable_reset_opacity = enable_reset_opacity
        self.enable_save_rendered_images = enable_save_rendered_images
        self.enable_save_rendered_depth = enable_save_rendered_depth
        self.enable_save_rendered_normals = enable_save_rendered_normals
        self.enable_save_rendered_alpha = enable_save_rendered_alpha
        self.enable_save_rendered_extra_attrs = enable_save_rendered_extra_attrs
        self.enable_save_skybox = enable_save_skybox
        self.cam_update_from_iter = cam_update_from_iter
        self.densify_from_iter = densify_from_iter
        self.densification_interval = densification_interval
        self.opacity_reset_interval = opacity_reset_interval
        self.sem_ignore_index = sem_ignore_index
        self.densify_grad_threshold = densify_grad_threshold
        self.densify_until_iter = densify_until_iter
        self.opacity_reset_until_iter = opacity_reset_until_iter
        self.iterations = iterations

        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.eval_rate = eval_rate
        self.save_dir = save_dir
        self.sh_increase_interval = sh_increase_interval
        self.background = torch.tensor(bg_color, dtype=torch.float32, device=self.device)
        self.loss_weights = loss_weights
        self.last_metrics = None
        self._callbacks_closed = False
        self._is_training = False
        self.loss_computer = GaussianLossComputer(
            weights=loss_weights,
            device=self.device,
            sem_ignore_index=sem_ignore_index,
            use_sift_loss=use_sift_loss,
        )
        self.metric_computer = EvalMetricComputer(device=self.device)
        self.checkpoints = CheckpointManager(save_dir=save_dir, enable_save_skybox=enable_save_skybox)
        self.output_saver = RenderOutputSaver(
            save_dir=save_dir,
            save_images=enable_save_rendered_images,
            save_depth=enable_save_rendered_depth,
            save_normals=enable_save_rendered_normals,
            save_alpha=enable_save_rendered_alpha,
            save_extra_attrs=enable_save_rendered_extra_attrs,
        )
        self.run_config = self._collect_run_config()
        self.checkpoints.set_run_config(self.run_config)
        if callbacks is None:
            callbacks = default_callbacks(
                wandb_project=wandb_project,
                wandb_name=wandb_name,
                save_dir=save_dir,
                enable_progress=enable_progress,
            )
        if extra_callbacks is not None:
            callbacks = list(callbacks) + list(extra_callbacks)
        self.callbacks = CallbackList(callbacks)
        self.callbacks.setup(self)

    def train(self):
        indices_random = self._reshuffle_train_indices()
        self.last_metrics = None
        self._is_training = True
        self.callbacks.on_train_start(self)

        try:
            for iteration in range(self.iterations):
                self._before_train_iteration(iteration)

                if len(indices_random) == 0:
                    indices_random = self._reshuffle_train_indices()

                viewpoint_cam = self.cams[indices_random.pop()]
                render_pkg = render(viewpoint_cam, self.gaussians, self.background)
                loss, loss_scalars = self.loss_computer(
                    render_pkg,
                    viewpoint_cam,
                    enable_cam_update=self.enable_cam_update,
                )

                loss.backward()
                viewspace_point_tensor_grad = render_pkg["viewspace_points"].grad
                self._optimizer_step(iteration)

                self.callbacks.on_train_iteration_end(self, iteration, loss_scalars, render_pkg)
                self._densification_step(iteration, viewspace_point_tensor_grad, render_pkg["visibility_filter"])
        finally:
            self._is_training = False
            self.close()

    def close(self):
        if self._callbacks_closed:
            return
        self.callbacks.on_train_end(self)
        self._callbacks_closed = True

    def _reshuffle_train_indices(self):
        indices = deepcopy(self.indices_for_train)
        random.shuffle(indices)
        return indices

    def _before_train_iteration(self, iteration):
        if self.enable_freeze is False:
            self.gaussians.update_learning_rate(iteration)
        if (iteration + 1) % self.sh_increase_interval == 0:
            self.gaussians.oneupSHdegree()

    def _optimizer_step(self, iteration):
        if self.enable_freeze is False:
            self.gaussians.optimizer.step()
            self.gaussians.optimizer.zero_grad(set_to_none=True)
        if self.gaussians.skyboxer is not None:
            self.gaussians.skyboxer.optimizer.step()
            self.gaussians.skyboxer.optimizer.zero_grad(set_to_none=True)
        if self.optimizer_cam is not None and iteration >= self.cam_update_from_iter:
            self.optimizer_cam.step()
            self.optimizer_cam.zero_grad(set_to_none=True)
            self.scheduler_cam.step()

    def _densification_step(self, iteration, viewspace_point_tensor_grad, visibility_filter):
        with torch.no_grad():
            if iteration < self.densify_until_iter and self.enable_densification:
                if viewspace_point_tensor_grad is not None and self.gaussians.optimizer is not None:
                    viewspace_point_tensor_grad = viewspace_point_tensor_grad[self.gaussians.num_fixed_points:]
                    visibility_filter = visibility_filter[self.gaussians.num_fixed_points:].nonzero()
                    self.gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)
                    if iteration > self.densify_from_iter and (iteration + 1) % self.densification_interval == 0:
                        self.gaussians.densify_and_prune(self.densify_grad_threshold, self.min_opacity, self.N_split)

            should_reset_opacity = (
                (iteration + 1) % self.opacity_reset_interval == 0
                and iteration < self.opacity_reset_until_iter
                and self.enable_reset_opacity
            )
            if should_reset_opacity:
                self.gaussians.reset_opacity()

    def should_eval(self, iteration):
        return (iteration + 1) % self.eval_interval == 0 or iteration == self.iterations - 1 or iteration == 0

    def should_save(self, iteration):
        return (iteration + 1) % self.save_interval == 0 or iteration == self.iterations - 1

    @torch.no_grad()
    def eval(self, iteration=0):
        self.callbacks.on_eval_start(self, iteration, len(self.indices_for_eval))
        records = []
        for idx, ind_for_eval in enumerate(self.indices_for_eval):
            viewpoint_cam = self.cams[ind_for_eval]
            render_pkg = render(viewpoint_cam, self.gaussians, self.background)
            self.output_saver.save(render_pkg, viewpoint_cam, iteration)
            metrics = self.metric_computer(render_pkg, viewpoint_cam)
            records.append(metrics)
            self.callbacks.on_eval_batch_end(self, iteration, idx, metrics)

        means = {
            key: sum(record[key] for record in records) / len(records)
            for key in records[0]
        }
        self.callbacks.on_eval_end(self, iteration, means)
        return means

    def save_outputs(self, render_pkg, viewpoint_cam, iteration):
        self.output_saver.save(render_pkg, viewpoint_cam, iteration)

    def save_colmap(self, save_dir, save_image=False):
        save_colmap_reconstruction(self.gaussians, self.cams, save_dir, save_image=save_image)
