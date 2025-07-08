#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.cameras import Camera


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics,Height,Width,images_folder=None,depths_folder=None, normals_folder=None, alphas_folder=None, extra_attrs_folder=None,preload=True):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        extr_name = extr.name
        height = intr.height
        width = intr.width
        scale_height = 1.0
        scale_width = 1.0
        if Height is not None and Width is not None:
            scale_height = Height / height
            scale_width = Width / width
            height = Height
            width = Width
        if os.path.exists(extr_name):
            image_name = os.path.basename(extr_name)
            image_path = extr_name
        else:
            image_name =os.path.basename(extr_name)
            image_path =os.path.join(images_folder, image_name)
            
        
        image = image_path
        endwith = image_name.split('.')[-1]
        depth = os.path.join(depths_folder, image_name.replace(endwith, "npy")) if depths_folder else None
        normal = os.path.join(normals_folder, image_name.replace(endwith, "npy")) if normals_folder else None
        alpha = os.path.join(alphas_folder, image_name.replace(endwith, "png")) if alphas_folder else None
        extra_attrs = os.path.join(extra_attrs_folder, image_name.replace(endwith, "npy")) if extra_attrs_folder else None

        R = qvec2rotmat(extr.qvec)
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]* scale_width
            focal_length_y = focal_length_x* scale_height
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]* scale_width
            focal_length_y = intr.params[1]* scale_height
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        
        cam_info = Camera(idx=idx, R=R, T=T, FoVx=FovX, FoVy=FovY,focal_length_x=focal_length_x,focal_length_y=focal_length_y,height=height,width=width,image=image,depth=depth,
                          normal=normal, alpha=alpha, extra_attrs=extra_attrs,
                            image_path=image_path, image_name=image_name,preload=preload)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos
def readCameras(cam_extrinsics, cam_intrinsics,Height,Width,images):
    cam_infos = []
    if images is None:
        images = np.zeros((len(cam_extrinsics), Height, Width, 3), dtype=np.uint8)  # Placeholder if images not provided
    for idx in range(len(cam_extrinsics)):
        
        
        intr = cam_intrinsics[idx]
        extr = cam_extrinsics[idx]
        R = extr[:3, :3]  # Assuming extr is a 4x4 matrix
        T = extr[:3, 3]  # Extract translation vector from extr
      
        image = images[idx]#(H,W,3)
        if isinstance(image, str):
            image_path = image
            image_name = os.path.basename(image_path)
            image = Image.open(image)
            width, height = image.size
        elif isinstance(image, np.ndarray):
            image_name=f"{idx:06d}.png"
            image_path = image_name
            image = image.astype(np.uint8)
            image = Image.fromarray(image)
            width, height = image.size
        elif isinstance(image, Image.Image):
            image_name = f"{idx:06d}.png"
            image_path = image_name
            width, height = image.size
        else:
            raise TypeError("Unsupported image type: {}".format(type(image)))
        scale_width = 1.0
        scale_height = 1.0
        if Height is not None and Width is not None:
            scale_height = Height / height
            scale_width = Width / width
            height = Height
            width = Width
            image = image.resize((Width, Height), Image.LANCZOS)

        focal_length_x = intr[0, 0]* scale_width
        focal_length_y = intr[1, 1] * scale_height
        FovX = focal2fov(focal_length_x, width)
        FovY = focal2fov(focal_length_y, height)
        cam_info = Camera(idx=idx, R=R, T=T, FoVx=FovX, FoVy=FovY,focal_length_x=focal_length_x,focal_length_y=focal_length_y,height=height,width=width,image=image,depth=None,
                          normal=None, alpha=None, extra_attrs=None,
                            image_path=image_path, image_name=image_name,preload=True)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos
# def readCamerasFromTransforms(path, transformsfile, depths_folder, white_background, is_test, extension=".png"):
#     cam_infos = []

#     with open(os.path.join(path, transformsfile)) as json_file:
#         contents = json.load(json_file)
#         fovx = contents["camera_angle_x"]

#         frames = contents["frames"]
#         for idx, frame in enumerate(frames):
#             cam_name = os.path.join(path, frame["file_path"] + extension)

#             # NeRF 'transform_matrix' is a camera-to-world transform
#             c2w = np.array(frame["transform_matrix"])
#             # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
#             c2w[:3, 1:3] *= -1

#             # get the world-to-camera transform and set R, T
#             w2c = np.linalg.inv(c2w)
#             R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
#             T = w2c[:3, 3]

#             image_path = os.path.join(path, cam_name)
#             image_name = Path(cam_name).stem
#             image = Image.open(image_path)

#             im_data = np.array(image.convert("RGBA"))

#             bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

#             norm_data = im_data / 255.0
#             arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
#             image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

#             fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
#             FovY = fovy 
#             FovX = fovx

