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
import math
from diff_gauss_pose import GaussianRasterizer, GaussianRasterizationSettings
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, bg_color : torch.Tensor, scaling_modifier = 1.0):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        projmatrix=viewpoint_camera.projection_matrix,
        sh_degree=pc.active_sh_degree,
        prefiltered=False,
        debug=False,
        enable_cov_grad= True,
        enable_sh_grad=True,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

   
    scales = pc.get_scaling
    rotations = pc.get_rotation

    
    shs = None
    colors_precomp = None
    shs = pc.get_features
    
    extra_attrs = pc.get_extra_attrs
    if pc._extra_attrs_dim==0:
        extra_attrs = None
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    
    rendered_image, rendered_depth, rendered_norm, rendered_alpha, radii, extra_attrs  = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3Ds_precomp = cov3D_precomp,
            extra_attrs = extra_attrs,
            viewmatrix = viewpoint_camera.world_view_transform,)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : radii > 0,
        "radii": radii,
        "depth" : rendered_depth[0],
        "normals" : rendered_norm,
        "alpha" : rendered_alpha[0],
        "extra_attrs" : extra_attrs,
        }
    
    return out
