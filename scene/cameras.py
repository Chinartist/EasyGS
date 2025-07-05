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

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View, getProjectionMatrix
import cv2
from PIL import Image
from torchvision.transforms.functional import to_tensor, resize
from scipy.spatial.transform import Rotation
def qvec2rotmat(qvec):
    rotmat = torch.zeros((3, 3), dtype=torch.float32, device=qvec.device)
    rotmat[0, 0] = 1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2
    rotmat[0, 1] = 2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3]
    rotmat[0, 2] = 2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
    rotmat[1, 0] = 2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3]
    rotmat[1, 1] = 1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2
    rotmat[1, 2] = 2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
    rotmat[2, 0] = 2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2]
    rotmat[2, 1] = 2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1]
    rotmat[2, 2] = 1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2
    return rotmat
def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec
class Camera(nn.Module):
    def __init__(self, idx, R, T, FoVx, FoVy,focal_length_x,focal_length_y,image, depth,normal,alpha,extra_attrs,
                 image_path,
                 image_name,
                 ):
        super(Camera, self).__init__()

        self.idx=idx
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.focal_length_x = focal_length_x
        self.focal_length_y = focal_length_y
        self.image_name = image_name
        self.image_path = image_path    
        self.image_gt = to_tensor(image)
        if depth is not None:
            self.depth_gt = torch.tensor(np.array(depth), dtype=torch.float32)
        else:
            self.depth_gt = None
        if normal is not None:
            self.normal_gt = torch.tensor(np.array(normal), dtype=torch.float32)
        else:
            self.normal_gt = None
        if alpha is not None:
            self.alpha_gt = torch.tensor(np.array(alpha), dtype=torch.float32)
        else:
            self.alpha_gt = None
        if extra_attrs is not None:
            self.extra_attrs_gt = torch.tensor(np.array(extra_attrs), dtype=torch.int16)
        else:
            self.extra_attrs_gt = None

        self.image_width = self.image_gt.shape[2]
        self.image_height = self.image_gt.shape[1]
        self.zfar = 100.0
        self.znear = 0.01
        # self.world_view_transform = torch.tensor(getWorld2View(R, T)).transpose(0, 1).cuda()#w2c.T
        self.qvec  = nn.Parameter(torch.tensor(rotmat2qvec(R)), requires_grad=True)
        self.tvec = nn.Parameter(torch.tensor(T, dtype=torch.float32), requires_grad=True)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()#c2image.T 
        self.cuda()
    @property
    def world_view_transform(self):
        R = qvec2rotmat(torch.nn.functional.normalize(self.qvec,dim=0))
        w2c = torch.zeros((4, 4), dtype=torch.float32, device=self.qvec.device)
        w2c[:3, :3] = R
        w2c[:3, 3] = self.tvec
        w2c[3, 3] = 1.0
        return w2c.transpose(0, 1)