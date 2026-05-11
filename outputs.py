from __future__ import annotations

import os

import matplotlib
import numpy as np
from PIL import Image

CMAP = matplotlib.colormaps.get_cmap("Spectral_r")


class RenderOutputSaver:
    def __init__(
        self,
        save_dir: str,
        save_images: bool = True,
        save_depth: bool = False,
        save_normals: bool = False,
        save_alpha: bool = False,
        save_extra_attrs: bool = False,
    ):
        self.save_dir = save_dir
        self.save_images = save_images
        self.save_depth = save_depth
        self.save_normals = save_normals
        self.save_alpha = save_alpha
        self.save_extra_attrs = save_extra_attrs

    def save(self, render_pkg, camera, iteration: int) -> None:
        image_name = os.path.splitext(camera.image_name)[0]

        if self.save_images:
            self._save_rgb(render_pkg["render"], iteration, image_name)
        if self.save_depth:
            self._save_depth(render_pkg["depth"], iteration, image_name)
        if self.save_normals:
            self._save_npy("rendered_normals", render_pkg["normals"], iteration, image_name)
        if self.save_alpha:
            self._save_npy("rendered_alpha", render_pkg["alpha"], iteration, image_name)
        if self.save_extra_attrs:
            self._save_npy("rendered_extra_attrs", render_pkg["extra_attrs"], iteration, image_name)

    def _save_rgb(self, rgb, iteration: int, image_name: str) -> None:
        save_path = self._path(iteration, "rendered_images", image_name, ".png")
        rgb = (rgb.detach() * 255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        Image.fromarray(rgb).save(save_path)

    def _save_depth(self, depth, iteration: int, image_name: str) -> None:
        raw_path = self._path(iteration, "rendered_depth", image_name, ".npy")
        depth = depth.float().detach().cpu().numpy()
        np.save(raw_path, depth)

        color_path = self._path(iteration, "rendered_depth_wcolor", image_name, ".png")
        depth_range = depth.max() - depth.min()
        if depth_range <= 1e-12:
            color_depth = np.zeros_like(depth)
        else:
            color_depth = (depth - depth.min()) / depth_range * 255.0
        color_depth = (CMAP(color_depth.astype(np.int32))[:, :, :3])[:, :, ::-1]
        Image.fromarray((color_depth * 255).astype(np.uint8)).save(color_path)

    def _save_npy(self, folder: str, tensor, iteration: int, image_name: str) -> None:
        if tensor is None:
            return
        save_path = self._path(iteration, folder, image_name, ".npy")
        np.save(save_path, tensor.float().detach().cpu().numpy())

    def _path(self, iteration: int, folder: str, image_name: str, suffix: str) -> str:
        directory = os.path.join(self.save_dir, f"{iteration}", folder)
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, f"{image_name}{suffix}")
