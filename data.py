from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from torch import nn

if __package__:
    from .scene.colmap_loader import (
        read_extrinsics_binary,
        read_extrinsics_text,
        read_intrinsics_binary,
        read_intrinsics_text,
        read_points3D_binary,
        read_points3D_text,
    )
    from .scene.dataset_readers import getNerfppNorm, readCameras, readColmapCameras
    from .scene.gaussian_model import GaussianModel
else:
    from scene.colmap_loader import (
        read_extrinsics_binary,
        read_extrinsics_text,
        read_intrinsics_binary,
        read_intrinsics_text,
        read_points3D_binary,
        read_points3D_text,
    )
    from scene.dataset_readers import getNerfppNorm, readCameras, readColmapCameras
    from scene.gaussian_model import GaussianModel


@dataclass
class SceneBuildResult:
    gaussians: GaussianModel
    cameras: nn.ModuleList


def build_scene(
    *,
    colmap_path=None,
    w2c=None,
    intrinsics=None,
    images=None,
    xyz=None,
    rgb=None,
    pretrained_path=None,
    height=None,
    width=None,
    images_folder=None,
    depths_folder=None,
    normals_folder=None,
    alphas_folder=None,
    extra_attrs_folder=None,
    preload=True,
    init_degree=0,
    max_sh_degree=3,
    extra_attrs_dim=0,
    percent_dense=0.01,
    verbose=True,
    add_skybox=False,
    skybox_points=100_000,
    skybox_radius_scale=10.0,
) -> SceneBuildResult:
    gaussians = GaussianModel(init_degree, max_sh_degree, extra_attrs_dim, percent_dense, verbose=verbose)

    if colmap_path is not None:
        return _build_from_colmap(
            gaussians=gaussians,
            colmap_path=colmap_path,
            pretrained_path=pretrained_path,
            height=height,
            width=width,
            images_folder=images_folder,
            depths_folder=depths_folder,
            normals_folder=normals_folder,
            alphas_folder=alphas_folder,
            extra_attrs_folder=extra_attrs_folder,
            preload=preload,
            add_skybox=add_skybox,
            skybox_points=skybox_points,
            skybox_radius_scale=skybox_radius_scale,
        )

    if _has_array_scene_inputs(w2c, intrinsics, images, xyz, pretrained_path, height, width):
        return _build_from_arrays(
            gaussians=gaussians,
            w2c=w2c,
            intrinsics=intrinsics,
            images=images,
            xyz=xyz,
            rgb=rgb,
            height=height,
            width=width,
            add_skybox=add_skybox,
            skybox_points=skybox_points,
            skybox_radius_scale=skybox_radius_scale,
        )

    raise ValueError("Either colmap_path or w2c, intrinsics, images and (xyz or pretrained_path) must be provided.")


def _build_from_colmap(
    *,
    gaussians,
    colmap_path,
    pretrained_path,
    height,
    width,
    images_folder,
    depths_folder,
    normals_folder,
    alphas_folder,
    extra_attrs_folder,
    preload,
    add_skybox,
    skybox_points,
    skybox_radius_scale,
) -> SceneBuildResult:
    print(f"Using COLMAP path: {colmap_path}")
    cam_extrinsics, cam_intrinsics = _read_colmap_cameras(colmap_path)
    if images_folder is None:
        images_folder = os.path.join(colmap_path, "images/")

    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=images_folder,
        depths_folder=depths_folder,
        normals_folder=normals_folder,
        alphas_folder=alphas_folder,
        extra_attrs_folder=extra_attrs_folder,
        Height=height,
        Width=width,
        preload=preload,
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)
    nerf_normalization = getNerfppNorm(cam_infos)
    xyz, rgb = _read_colmap_points(colmap_path)

    if xyz is not None:
        _initialize_gaussians_from_points(
            gaussians,
            xyz,
            rgb,
            nerf_normalization,
            len(cam_infos),
            add_skybox,
            skybox_points,
            skybox_radius_scale,
        )
    elif pretrained_path is None:
        raise FileNotFoundError(
            "COLMAP points3D file was not found. Provide sparse/0/points3D.bin, "
            "sparse/0/points3D.txt, or a pretrained_path."
        )
    else:
        gaussians.scene_extent = nerf_normalization["radius"]

    return SceneBuildResult(gaussians=gaussians, cameras=nn.ModuleList(cam_infos))


def _build_from_arrays(
    *,
    gaussians,
    w2c,
    intrinsics,
    images,
    xyz,
    rgb,
    height,
    width,
    add_skybox,
    skybox_points,
    skybox_radius_scale,
) -> SceneBuildResult:
    print("Using provided camera parameters.")
    cam_infos_unsorted = readCameras(cam_extrinsics=w2c, cam_intrinsics=intrinsics, images=images, Height=height, Width=width)
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)
    nerf_normalization = getNerfppNorm(cam_infos)

    if xyz is not None:
        if rgb is None:
            rgb = np.ones_like(xyz, dtype=np.float32) * 0.5
        _initialize_gaussians_from_points(
            gaussians,
            xyz,
            rgb,
            nerf_normalization,
            len(cam_infos),
            add_skybox,
            skybox_points,
            skybox_radius_scale,
        )
    else:
        gaussians.scene_extent = nerf_normalization["radius"]

    return SceneBuildResult(gaussians=gaussians, cameras=nn.ModuleList(cam_infos))


def _read_colmap_cameras(colmap_path):
    sparse_dir = os.path.join(colmap_path, "sparse/0")
    images_bin = os.path.join(sparse_dir, "images.bin")
    cameras_bin = os.path.join(sparse_dir, "cameras.bin")
    if os.path.exists(images_bin) and os.path.exists(cameras_bin):
        return read_extrinsics_binary(images_bin), read_intrinsics_binary(cameras_bin)

    images_txt = os.path.join(sparse_dir, "images.txt")
    cameras_txt = os.path.join(sparse_dir, "cameras.txt")
    if not os.path.exists(images_txt) or not os.path.exists(cameras_txt):
        raise FileNotFoundError(f"COLMAP cameras not found under {sparse_dir}")
    return read_extrinsics_text(images_txt), read_intrinsics_text(cameras_txt)


def _read_colmap_points(colmap_path):
    bin_path = os.path.join(colmap_path, "sparse/0/points3D.bin")
    txt_path = os.path.join(colmap_path, "sparse/0/points3D.txt")
    if os.path.exists(bin_path):
        xyz, rgb, _ = read_points3D_binary(bin_path)
        return xyz, rgb
    if os.path.exists(txt_path):
        xyz, rgb, _ = read_points3D_text(txt_path)
        return xyz, rgb
    return None, None


def _initialize_gaussians_from_points(
    gaussians,
    xyz,
    rgb,
    nerf_normalization,
    camera_count: int,
    add_skybox: bool,
    skybox_points: int,
    skybox_radius_scale: float,
) -> None:
    rgb = _normalize_rgb(rgb, xyz)
    if camera_count == 1:
        nerf_normalization["radius"] = np.max(np.linalg.norm(xyz - np.mean(xyz, axis=0), axis=1)) * 1.1
    gaussians.create_from_pcd(
        xyz,
        rgb,
        nerf_normalization["radius"],
        add_skybox,
        skybox_points=skybox_points,
        skybox_radius_scale=skybox_radius_scale,
    )


def _normalize_rgb(rgb, xyz):
    if rgb is None:
        return np.ones_like(xyz, dtype=np.float32) * 0.5
    if rgb.max() > 1.0:
        return rgb / 255.0
    return rgb


def _has_array_scene_inputs(w2c, intrinsics, images, xyz, pretrained_path, height, width) -> bool:
    has_images_or_size = images is not None or (height is not None and width is not None)
    has_geometry = xyz is not None or pretrained_path is not None
    return w2c is not None and intrinsics is not None and has_images_or_size and has_geometry
