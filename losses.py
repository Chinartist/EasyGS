from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

if __package__:
    from .utils.image_utils import psnr
    from .utils.loss_utils import l1_loss, reprojection_loss, ssim
else:
    from utils.image_utils import psnr
    from utils.loss_utils import l1_loss, reprojection_loss, ssim


def composite_with_alpha(rendered: torch.Tensor, target: torch.Tensor, alpha: Optional[torch.Tensor]) -> torch.Tensor:
    if alpha is None:
        return rendered
    return rendered * (1 - alpha[None]) + target * alpha[None]


class GaussianLossComputer:
    def __init__(
        self,
        weights: Dict[str, float],
        device: torch.device,
        sem_ignore_index: int = 255,
        use_sift_loss: bool = False,
    ):
        self.weights = weights
        self.device = device
        self.sem_ignore_index = sem_ignore_index
        self.use_sift_loss = use_sift_loss

    def __call__(self, render_pkg, camera, enable_cam_update: bool = False) -> Tuple[torch.Tensor, Dict[str, float]]:
        image = render_pkg["render"]
        depth = render_pkg["depth"]
        normals = render_pkg["normals"]
        alpha = render_pkg["alpha"]
        extra_attrs = render_pkg["extra_attrs"]

        image_gt = camera.get_image_gt.to(self.device)
        depth_gt = self._to_device(camera.get_depth_gt)
        normal_gt = self._to_device(camera.get_normal_gt)
        alpha_gt = self._to_device(camera.get_alpha_gt)
        extra_attrs_gt = self._to_device(camera.get_extra_attrs_gt)

        image = composite_with_alpha(image, image_gt, alpha_gt)

        rgb_l1 = l1_loss(image, image_gt)
        rgb_ssim = ssim(image, image_gt)
        total = (
            rgb_l1 * self.weights["rgb_l1_weight"]
            + (1.0 - rgb_ssim) * self.weights["rgb_ssim_weight"]
        )
        scalars = {
            "rgb_l1": rgb_l1.item(),
            "rgb_ssim": rgb_ssim.item(),
        }

        if enable_cam_update and self.use_sift_loss:
            sift_loss = reprojection_loss(image, image_gt, device=str(self.device))
            total = total + sift_loss * self.weights["sift_weight"]
            scalars["sift"] = sift_loss.item()

        if depth_gt is not None:
            depth_loss = F.mse_loss(depth, depth_gt)
            total = total + depth_loss * self.weights["depth_weight"]
            scalars["depth"] = depth_loss.item()

        if normal_gt is not None:
            normal_loss = F.mse_loss(normals, normal_gt)
            total = total + normal_loss * self.weights["normals_weight"]
            scalars["normals"] = normal_loss.item()

        if alpha_gt is not None:
            alpha_loss = (alpha_gt * alpha).mean()
            total = total + alpha_loss * self.weights["alpha_weight"]
            scalars["alpha"] = alpha_loss.item()

        if extra_attrs_gt is not None and extra_attrs is not None:
            extra_attrs_loss = F.cross_entropy(
                extra_attrs[None],
                extra_attrs_gt[None],
                ignore_index=self.sem_ignore_index,
            )
            total = total + extra_attrs_loss * self.weights["extra_attrs_weight"]
            scalars["extra_attrs"] = extra_attrs_loss.item()

        scalars["total"] = total.item()
        return total, scalars

    def _to_device(self, value):
        return value.to(self.device) if value is not None else None


class EvalMetricComputer:
    def __init__(self, device: torch.device):
        self.device = device

    def __call__(self, render_pkg, camera) -> Dict[str, float]:
        image = render_pkg["render"]
        image_gt = camera.get_image_gt.to(self.device)
        alpha_gt = camera.get_alpha_gt
        if alpha_gt is not None:
            alpha_gt = alpha_gt.to(self.device)
        image = composite_with_alpha(image, image_gt, alpha_gt)

        return {
            "l1": l1_loss(image, image_gt).item(),
            "ssim": ssim(image, image_gt).item(),
            "psnr": psnr(image, image_gt).mean().item(),
        }
