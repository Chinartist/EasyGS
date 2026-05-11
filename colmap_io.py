from __future__ import annotations

import os

import numpy as np
import pycolmap
from PIL import Image


def build_pycolmap_intrinsics(fidx, intrinsics, camera_type, extra_params=None):
    if camera_type == "PINHOLE":
        return np.array(
            [intrinsics[fidx][0, 0], intrinsics[fidx][1, 1], intrinsics[fidx][0, 2], intrinsics[fidx][1, 2]]
        )
    if camera_type == "SIMPLE_PINHOLE":
        focal = (intrinsics[fidx][0, 0] + intrinsics[fidx][1, 1]) / 2
        return np.array([focal, intrinsics[fidx][0, 2], intrinsics[fidx][1, 2]])
    if camera_type == "SIMPLE_RADIAL":
        raise NotImplementedError("SIMPLE_RADIAL is not supported yet")
    raise ValueError(f"Camera type {camera_type} is not supported yet")


def batch_np_matrix_to_pycolmap_wo_track(
    points3d,
    points_rgb,
    extrinsics,
    intrinsics,
    image_names,
    image_size,
    shared_camera=False,
    camera_type="PINHOLE",
):
    reconstruction = pycolmap.Reconstruction()
    if points_rgb is None:
        points_rgb = np.ones((len(points3d), 3)) * 128

    for vidx, point in enumerate(points3d):
        reconstruction.add_point3D(point, pycolmap.Track(), points_rgb[vidx])

    camera = None
    for fidx, extrinsic in enumerate(extrinsics):
        if camera is None or not shared_camera:
            camera = pycolmap.Camera(
                model=camera_type,
                width=image_size[0],
                height=image_size[1],
                params=build_pycolmap_intrinsics(fidx, intrinsics, camera_type),
                camera_id=fidx + 1,
            )
            reconstruction.add_camera(camera)

        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(extrinsic[:3, :3]),
            extrinsic[:3, 3],
        )
        reconstruction.add_image(
            pycolmap.Image(
                id=fidx + 1,
                name=image_names[fidx],
                camera_id=camera.camera_id,
                cam_from_world=cam_from_world,
            )
        )

    return reconstruction


def save_colmap_reconstruction(gaussians, cameras, save_dir, save_image=False):
    intrinsics = []
    extrinsics = []
    image_names = [cam.image_name for cam in cameras]

    os.makedirs(os.path.join(save_dir, "sparse/0"), exist_ok=True)
    if save_image:
        os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
        print(f"Saving images to {os.path.join(save_dir, 'images')}")

    for cam in cameras:
        intr = np.zeros((3, 3), dtype=np.float32)
        intr[0, 0] = cam.focal_length_x
        intr[1, 1] = cam.focal_length_y
        intr[0, 2] = cam.principal_point_x
        intr[1, 2] = cam.principal_point_y
        intrinsics.append(intr)

        extrinsics.append(cam.world_view_transform.detach().cpu().transpose(0, 1).numpy()[:3, :4])

        if save_image:
            image = (cam.get_image_gt * 255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            Image.fromarray(image).save(os.path.join(save_dir, "images", f"{cam.image_name}"))

    print(f"Saving COLMAP data to {save_dir}")
    reconstruction = batch_np_matrix_to_pycolmap_wo_track(
        points3d=gaussians._xyz.detach().cpu().numpy(),
        points_rgb=None,
        extrinsics=np.stack(extrinsics, axis=0),
        intrinsics=np.stack(intrinsics, axis=0),
        image_names=image_names,
        image_size=(cameras[0].image_width, cameras[0].image_height),
        shared_camera=False,
        camera_type="PINHOLE",
    )
    reconstruction.write_text(os.path.join(save_dir, "sparse/0/"))
