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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.general_utils import strip_symmetric, build_scaling_rotation


class GaussianModel():

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.opacity_inverse_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self,init_sh_dgree, max_sh_degree,extra_attrs_dim=0,percent_dense=0.01,verbose=True):
        assert init_sh_dgree <= max_sh_degree, "Initial SH degree must be less than or equal to the maximum SH degree."
        self.active_sh_degree = init_sh_dgree
        self.max_sh_degree = max_sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)

        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self._extra_attrs_dim = extra_attrs_dim
   
        self.percent_dense = percent_dense
        self.scene_extent = 1.0
        self.skyboxer = None
        self.num_fixed_points =0
        self.verbose = verbose
        self.setup_functions()
    @property
    def get_scaling(self):
        if self.skyboxer is not None:
            return torch.cat([self.skyboxer.get_scaling, self.scaling_activation(self._scaling)], dim=0)
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        if self.skyboxer is not None:
            return torch.cat([self.skyboxer.get_rotation, self.rotation_activation(self._rotation)], dim=0)
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        if self.skyboxer is not None:
            return torch.cat([self.skyboxer.get_xyz, self._xyz], dim=0)
        return self._xyz
    
    @property
    def get_extra_attrs(self):
        if self.skyboxer is not None:
            return torch.cat([self.skyboxer.get_extra_attrs, self._extra_attrs], dim=0)
       
        return self._extra_attrs
       

    @property
    def get_features(self):
        if self.skyboxer is not None:
            features_dc = torch.cat((self.skyboxer._features_dc, self._features_dc), dim=0)
            features_rest = torch.cat((self.skyboxer._features_rest, self._features_rest), dim=0)
            return torch.cat([features_dc, features_rest], dim=1)
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        if self.skyboxer is not None:
            return torch.cat([self.skyboxer._features_dc, self._features_dc], dim=0)
        return self._features_dc
    
    @property
    def get_features_rest(self):
        if self.skyboxer is not None:
            return torch.cat([self.skyboxer._features_rest, self._features_rest], dim=0)
        return self._features_rest
    
    @property
    def get_opacity(self):
        if self.skyboxer is not None:
            return torch.cat([self.skyboxer.get_opacity, self.opacity_activation(self._opacity)], dim=0)
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        if self.skyboxer is not None:
            scaling = torch.cat([self.skyboxer.get_scaling, self.get_scaling], dim=0)
            rotation = torch.cat([self.skyboxer._rotation, self._rotation], dim=0)
            return self.covariance_activation(scaling, scaling_modifier, rotation)
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self,xyz,rgb ,scene_extent : float,add_skybox = False,skybox_points = 100_000,skybox_radius_scale=2.0):
        self.scene_extent = scene_extent
        fused_point_cloud = torch.tensor(np.asarray(xyz)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(rgb)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
     
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(xyz)).float().cuda()), 0.0000001)
        scales = torch.log( torch.clamp_max(torch.sqrt(dist2),self.scene_extent*self.percent_dense) )[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.opacity_inverse_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        features_dc =features[:,:,0:1]
        features_rest =features[:,:,1:]
        if add_skybox:
            self.add_skybox(fused_point_cloud, skybox_points = skybox_points,skybox_radius_scale=skybox_radius_scale)
                       
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features_dc.transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features_rest.transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._extra_attrs = nn.Parameter(torch.zeros((fused_point_cloud.shape[0], self._extra_attrs_dim), device="cuda").requires_grad_(True))
 
           
    def add_skybox(self,xyz,skybox_points = 100_000,skybox_radius_scale=2.0):
            minimum, _ = torch.min(xyz, axis=0)
            maximum, _ = torch.max(xyz, axis=0)
            mean = 0.5 * (minimum + maximum)

            radius = torch.linalg.norm(maximum - mean)

            # xyz
            theta = (2.0 * torch.pi * torch.rand(skybox_points//2, device="cuda")).float()
            # phi = (torch.arccos(1.0 - 1.4 * torch.rand(skybox_points, device="cuda"))).float()
            phi = (torch.arccos( torch.rand(skybox_points//2, device="cuda"))).float()
            skybox_xyz = torch.zeros((skybox_points, 3))
            skybox_xyz[:skybox_points//2, 0] = radius * skybox_radius_scale * torch.cos(theta)*torch.sin(phi)
            skybox_xyz[:skybox_points//2, 1] = radius * skybox_radius_scale * torch.sin(theta)*torch.sin(phi)
            skybox_xyz[:skybox_points//2, 2] = -radius * skybox_radius_scale * torch.cos(phi)
            skybox_xyz[skybox_points//2:, 0] = radius * skybox_radius_scale * torch.cos(theta)*torch.sin(phi)
            skybox_xyz[skybox_points//2:, 1] = radius * skybox_radius_scale * torch.sin(theta)*torch.sin(phi)
            skybox_xyz[skybox_points//2:, 2] = radius * skybox_radius_scale * torch.cos(phi)
            skybox_xyz += mean.cpu()

            # sh
            fused_color = (torch.ones((skybox_points, 3)).cuda())
            fused_color[:skybox_points,0] *= (205/255)
            fused_color[:skybox_points,1] *= (218/255)
            fused_color[:skybox_points,2] *= (226/255)
            skyboxer =GaussianModel(init_sh_dgree=self.max_sh_degree, max_sh_degree=self.max_sh_degree,extra_attrs_dim=self._extra_attrs_dim,percent_dense=1e10)
            skyboxer.create_from_pcd(skybox_xyz.cpu().numpy(), fused_color.cpu().numpy(), scene_extent=self.scene_extent, add_skybox=False)
            self.num_fixed_points += skybox_points
            self.skyboxer = skyboxer
            
    def training_setup(self, lr
                       ):
 
        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': lr["position_lr_init"] * self.scene_extent, "name": "xyz"},
            {'params': [self._extra_attrs], 'lr': lr["extra_attrs_lr"], "name": "extra_attrs"},
            {'params': [self._features_dc], 'lr': lr["feature_lr"], "name": "f_dc"},
            {'params': [self._features_rest], 'lr': lr["feature_lr"] / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': lr["opacity_lr"], "name": "opacity"},
            {'params': [self._scaling], 'lr': lr["scaling_lr_init"], "name": "scaling"},
            {'params': [self._rotation], 'lr': lr["rotation_lr"], "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)


        self.xyz_scheduler_args = get_expon_lr_func(lr_init=lr['position_lr_init']*self.scene_extent,
                                                    lr_final=lr['position_lr_final']*self.scene_extent,
                                                    lr_delay_mult=lr['position_lr_delay_mult'],
                                                    max_steps=lr['position_lr_max_steps'])
        self.scaling_scheduler_args = get_expon_lr_func(lr_init=lr['scaling_lr_init'],
                                                    lr_final=lr['scaling_lr_final'],
                                                    lr_delay_mult=lr['scaling_lr_delay_mult'],
                                                    max_steps=lr['scaling_lr_max_steps'])

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
 

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
            if param_group["name"] == "scaling":
                lr = self.scaling_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path,enable_save_skybox=False):
        mkdir_p(os.path.dirname(path))
        if not enable_save_skybox:
            skyboxer = self.skyboxer
            self.skyboxer = None
        xyz = self.get_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self.get_features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self.get_features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self.opacity_inverse_activation(self.get_opacity).detach().cpu().numpy()
        scale = self.scaling_inverse_activation(self.get_scaling).detach().cpu().numpy()
        rotation = self.get_rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        if not enable_save_skybox:
            self.skyboxer = skyboxer

    def reset_opacity(self):
        opacities_new = self.opacity_inverse_activation(torch.min(self.opacity_activation(self._opacity), torch.ones_like(self.opacity_activation(self._opacity))*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
  
    def load_ply(self, path):
        plydata = PlyData.read(path)
    

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        # assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], 3*((self.max_sh_degree + 1) ** 2 - 1)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._extra_attrs = nn.Parameter(torch.zeros((xyz.shape[0], self._extra_attrs_dim), device="cuda").requires_grad_(True))
        
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._extra_attrs= optimizable_tensors["extra_attrs"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz,new_extra_attrs, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "extra_attrs": new_extra_attrs,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._extra_attrs = optimizable_tensors["extra_attrs"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")
    

    def densify_and_split(self, grads, grad_threshold, N=4):
        n_init_points = self._xyz.shape[0]
        # Extract points that satisfy the gradient condition
        #梯度过大 且 椭球scale小于 场景大小*percent_dense（0.01） -> 分裂一份，xyz取正态分布，scale变小0.8*N倍
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.scaling_activation(self._scaling), dim=1).values > self.percent_dense*self.scene_extent)

        stds = self.scaling_activation(self._scaling)[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self._xyz[selected_pts_mask].repeat(N, 1)
        new_extra_attrs = self._extra_attrs[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.scaling_activation(self._scaling)[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        if self.verbose:
            print(f"split {new_xyz.shape[0]} points ")
 
        self.densification_postfix(new_xyz,new_extra_attrs, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold):
        # Extract points that satisfy the gradient condition
        #梯度过大 且 椭球scale小于 场景大小*percent_dense（0.01） -> 复制一份
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.scaling_activation(self._scaling), dim=1).values <= self.percent_dense*self.scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_extra_attrs = self._extra_attrs[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        if self.verbose:
            print(f"clone {new_xyz.shape[0]} points ")
 
        self.densification_postfix(new_xyz,new_extra_attrs, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity,N):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad,)
        self.densify_and_split(grads, max_grad,N)
        
        #不透明度小于阈值以及scale大于self.percent_dense* scene_extent 均被删除
        prune_mask = (self.opacity_activation(self._opacity)< min_opacity).squeeze()
        
        # big_points_ws = self.scaling_activation(self._scaling).max(dim=1).values > self.percent_dense* self.scene_extent*10
        # prune_mask = torch.logical_or(prune_mask, big_points_ws)
        if self.verbose:
            print(f"prune {prune_mask.sum()} points \n")       
 
        self.prune_points(prune_mask)
    

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor_grad, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor_grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
