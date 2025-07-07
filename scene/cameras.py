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
    def __init__(self, idx, R, T, FoVx, FoVy,focal_length_x,focal_length_y,height,width,image, depth,normal,alpha,extra_attrs,
                 image_path,
                 image_name,preload=True
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
        if preload and isinstance(image,str):
            self.image_gt = to_tensor(Image.open(image).resize((width,height),Image.LANCZOS))
        else:
            self.image_gt = image
        
        if preload and isinstance(depth,str):
            self.depth_gt = torch.tensor(np.array(np.load(depth)), dtype=torch.float32)
            self.depth_gt = torch.nn.functional.interpolate(self.depth_gt[None,None],size=(height,width),mode="bilinear",align_corners=False)[0,0]
        else:
            self.depth_gt = depth
       
        
        if preload and isinstance(normal,str):
            self.normal_gt = torch.tensor(np.array(np.load(normal)), dtype=torch.float32)
            self.normal_gt = torch.nn.functional.interpolate(self.normal_gt[None],size=(height,width),mode="bilinear",align_corners=False)[0]
        else:
            self.normal_gt = normal
        
        
        if preload and isinstance(alpha,str):
            self.alpha_gt = torch.tensor(np.array(Image.open(alpha)), dtype=torch.float32)
            self.alpha_gt = torch.nn.functional.interpolate(self.alpha_gt[None,None],size=(height,width),mode="nearest")[0,0]
        else:
            self.alpha_gt = alpha
       
        
        if preload and isinstance(extra_attrs,str):
            self.extra_attrs_gt = torch.tensor(np.array(np.load(extra_attrs)), dtype=torch.int16)
            self.extra_attrs_gt = torch.nn.functional.interpolate(self.extra_attrs_gt[None,None].float(),size=(height,width),mode="nearest")[0,0].long()
        else:
            self.extra_attrs_gt = extra_attrs
      

        self.image_width = width
        self.image_height = height
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
    @property
    def get_image_gt(self):
        if isinstance(self.image_gt,str):
            return to_tensor(Image.open(self.image_gt).resize((self.image_width,self.image_height),Image.LANCZOS))
        else:
            return self.image_gt
    @property
    def get_depth_gt(self):
        if isinstance(self.depth_gt,str):
            return torch.nn.functional.interpolate(torch.tensor(np.array(np.load(self.depth_gt)), dtype=torch.float32)[None,None],size=(self.image_height,self.image_width),mode="bilinear",align_corners=False)[0,0]
        else:
            return self.depth_gt
    @property
    def get_normal_gt(self):
        if isinstance(self.normal_gt,str):
            return torch.nn.functional.interpolate(torch.tensor(np.array(np.load(self.normal_gt)), dtype=torch.float32)[None],size=(self.image_height,self.image_width),mode="bilinear",align_corners=False)[0]
        else:
            return self.normal_gt
    @property
    def get_alpha_gt(self):
        if isinstance(self.alpha_gt,str):
            alpha_gt = torch.nn.functional.interpolate(torch.tensor(np.array(Image.open(self.alpha_gt)), dtype=torch.float32)[None,None],size=(self.image_height,self.image_width),mode="nearest")[0,0]
            alpha_gt = (alpha_gt==0).float()
            return alpha_gt
        else:
            if self.alpha_gt is not None:
                return  (self.alpha_gt==0).float()
            return self.alpha_gt
    @property
    def get_extra_attrs_gt(self):
        if isinstance(self.extra_attrs_gt,str):
            return torch.nn.functional.interpolate(torch.tensor(np.array(np.load(self.extra_attrs_gt)), dtype=torch.int16)[None,None].float(),size=(self.image_height,self.image_width),mode="nearest")[0,0].long()
        else:
            return self.extra_attrs_gt