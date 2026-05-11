from __future__ import annotations

import json
import os
from typing import Dict, Optional

import torch
from torch import nn


def _torch_load(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


class CheckpointManager:
    def __init__(self, save_dir: str, enable_save_skybox: bool = False, run_config: Optional[Dict[str, object]] = None):
        self.save_dir = save_dir
        self.enable_save_skybox = enable_save_skybox
        self.run_config = run_config or {}

    def set_run_config(self, run_config: Dict[str, object]) -> None:
        self.run_config = run_config

    def save(self, iteration: int, gaussians, cameras, metrics: Optional[Dict[str, float]] = None) -> str:
        checkpoint_dir = os.path.join(self.save_dir, f"{iteration}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        gaussians.save_ply(os.path.join(checkpoint_dir, "model.ply"), self.enable_save_skybox)
        if gaussians._extra_attrs_dim > 0:
            gaussians.save_sem_ply(os.path.join(checkpoint_dir, "model_sem.ply"))

        torch.save(cameras.state_dict(), os.path.join(checkpoint_dir, "cameras.pth"))
        if self.enable_save_skybox:
            extra_attrs = gaussians.get_extra_attrs
            num_points = gaussians.get_xyz.shape[0]
        else:
            extra_attrs = gaussians._extra_attrs
            num_points = gaussians._xyz.shape[0]
        torch.save(extra_attrs.detach().cpu(), os.path.join(checkpoint_dir, "extra_attrs.pth"))

        metadata = {
            "iteration": iteration,
            "active_sh_degree": int(gaussians.active_sh_degree),
            "max_sh_degree": int(gaussians.max_sh_degree),
            "extra_attrs_dim": int(gaussians._extra_attrs_dim),
            "num_points": int(num_points),
            "metrics": metrics or {},
            "config": self.run_config,
        }
        with open(os.path.join(checkpoint_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return checkpoint_dir


def load_pretrained_checkpoint(pretrained_path: str, gaussians, cameras, device: torch.device) -> None:
    if pretrained_path.endswith(".ply"):
        print(f"Loading pretrained model from {pretrained_path}")
        gaussians.load_ply(pretrained_path)
        gaussians.active_sh_degree = gaussians.max_sh_degree
        return

    if not os.path.isdir(pretrained_path):
        raise ValueError("Pretrained path must be a directory or a .ply file.")

    print(f"Loading pretrained model from directory {pretrained_path}")
    metadata_path = os.path.join(pretrained_path, "metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        gaussians.max_sh_degree = int(metadata.get("max_sh_degree", gaussians.max_sh_degree))
        gaussians._extra_attrs_dim = int(metadata.get("extra_attrs_dim", gaussians._extra_attrs_dim))

    gaussians.load_ply(os.path.join(pretrained_path, "model.ply"))
    gaussians.active_sh_degree = min(
        int(metadata.get("active_sh_degree", gaussians.max_sh_degree)),
        gaussians.max_sh_degree,
    )

    cameras_path = os.path.join(pretrained_path, "cameras.pth")
    if os.path.exists(cameras_path):
        cameras.load_state_dict(_torch_load(cameras_path, device))

    extra_attrs_path = os.path.join(pretrained_path, "extra_attrs.pth")
    if os.path.exists(extra_attrs_path):
        extra_attrs = _torch_load(extra_attrs_path, device).to(device)
        if extra_attrs.shape[0] > gaussians._xyz.shape[0]:
            extra_attrs = extra_attrs[-gaussians._xyz.shape[0]:]
        if extra_attrs.shape[0] != gaussians._xyz.shape[0]:
            raise ValueError(
                "extra_attrs.pth point count does not match model.ply. "
                f"Got {extra_attrs.shape[0]} attrs for {gaussians._xyz.shape[0]} points."
            )
        gaussians._extra_attrs_dim = int(extra_attrs.shape[1]) if extra_attrs.ndim == 2 else 0
        gaussians._extra_attrs = nn.Parameter(
            extra_attrs,
            requires_grad=True,
        )
