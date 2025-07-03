
import sys
import os
sys.path.append( os.path.dirname(__file__) )
import torch
import random
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
import sys
from scene.gaussian_model import  GaussianModel
from utils.general_utils import  get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from scene.dataset_readers import readColmapCameras,readCameras, getNerfppNorm
from torch import nn
from PIL import Image
import numpy as np
import matplotlib
import pycolmap
from copy import deepcopy
from rich import print
from rich.progress import Progress,track, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn,SpinnerColumn ,RenderableColumn,MofNCompleteColumn

LearningRate =dict(
            position_lr_init = 0.00016,
            position_lr_final = 0.0000016,
            position_lr_delay_mult = 0.01,
            position_lr_max_steps = 30_000,
            cam_lr = 0.00016,
            extra_attrs_lr=0.001,
            feature_lr = 0.0025,
            opacity_lr = 0.025,
            scaling_lr = 0.005,
            rotation_lr = 0.001,
    )
LossWeights = dict(
            rgb_l1_weight = 0.8,
            rgb_ssim_weight = 0.2,
            depth_weight = 1.0,
            normals_weight = 1.0,
            alpha_weight = 1.0,
            extra_attrs_weight = 0.1,
    )
CMAP = matplotlib.colormaps.get_cmap('Spectral_r')
def _build_pycolmap_intri(fidx, intrinsics, camera_type, extra_params=None):
    """
    Helper function to get camera parameters based on camera type.

    Args:
        fidx: Frame index
        intrinsics: Camera intrinsic parameters
        camera_type: Type of camera model
        extra_params: Additional parameters for certain camera types

    Returns:
        pycolmap_intri: NumPy array of camera parameters
    """
    if camera_type == "PINHOLE":
        pycolmap_intri = np.array(
            [intrinsics[fidx][0, 0], intrinsics[fidx][1, 1], intrinsics[fidx][0, 2], intrinsics[fidx][1, 2]]
        )
    elif camera_type == "SIMPLE_PINHOLE":
        focal = (intrinsics[fidx][0, 0] + intrinsics[fidx][1, 1]) / 2
        pycolmap_intri = np.array([focal, intrinsics[fidx][0, 2], intrinsics[fidx][1, 2]])
    elif camera_type == "SIMPLE_RADIAL":
        raise NotImplementedError("SIMPLE_RADIAL is not supported yet")
        focal = (intrinsics[fidx][0, 0] + intrinsics[fidx][1, 1]) / 2
        pycolmap_intri = np.array([focal, intrinsics[fidx][0, 2], intrinsics[fidx][1, 2], extra_params[fidx][0]])
    else:
        raise ValueError(f"Camera type {camera_type} is not supported yet")

    return pycolmap_intri
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
    """
    Convert Batched NumPy Arrays to PyCOLMAP

    Different from batch_np_matrix_to_pycolmap, this function does not use tracks.

    It saves points3d to colmap reconstruction format only to serve as init for Gaussians or other nvs methods.

    Do NOT use this for BA.
    """
    # points3d: Px3
    # points_xyf: Px3, with x, y coordinates and frame indices
    # points_rgb: Px3, rgb colors
    # extrinsics: Nx3x4
    # intrinsics: Nx3x3
    # image_size: 2, assume all the frames have been padded to the same size
    # where N is the number of frames and P is the number of tracks

    N = len(extrinsics)
    P = len(points3d)

    # Reconstruction object, following the format of PyCOLMAP/COLMAP
    reconstruction = pycolmap.Reconstruction()
    if  points_rgb is None:
        points_rgb =np.ones((P, 3))* 128 #np.random.rand(P, 3)* 255
    for vidx in range(P):
        reconstruction.add_point3D(points3d[vidx], pycolmap.Track(), points_rgb[vidx])

    camera = None
    # frame idx
    for fidx in range(N):
        # set camera
        if camera is None or (not shared_camera):
            pycolmap_intri = _build_pycolmap_intri(fidx, intrinsics, camera_type)

            camera = pycolmap.Camera(
                model=camera_type, width=image_size[0], height=image_size[1], params=pycolmap_intri, camera_id=fidx + 1
            )

            # add camera
            reconstruction.add_camera(camera)
        # set image
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(extrinsics[fidx][:3, :3]), extrinsics[fidx][:3, 3]
        )  # Rot and Trans

        image = pycolmap.Image(
            id=fidx + 1, name=image_names[fidx], camera_id=camera.camera_id, cam_from_world=cam_from_world
        )
        # add image
        reconstruction.add_image(image)

    #保证在colmap gui中可见
    for vidx in range(P):
        # add track
        track = reconstruction.points3D[vidx+1].track
        track.add_element(1,1)
        track.add_element(1,1)
        track.add_element(1,1)
    return reconstruction
class GSTrainer():
    def __init__(self,
                    #要么提供COLMAP路径，要么提供相机参数
                    colmap_path=None,
                    w2c=None,
                    intrinsics=None,
                    images=None,
                    xyz=None,
                    rgb=None,
                    depths=None,
                    normals=None,
                    alphas=None,
                    extra_attrs=None,
                    pretrained_path=None,
                    save_dir=None,
                    height=None,
                    width=None,
                    #数据路径
                    images_folder=None,
                    depths_folder=None,
                    normals_folder=None,
                    alphas_folder=None,
                    extra_attrs_folder=None,
                    #训练和测试设置
                    add_skybox=False,
                    enable_densification=True,
                    enable_reset_opacity=True,
                    enable_train_all=True,
                    enable_cam_update=False,
                    

                    enable_save_rendered_images=True,
                    enable_save_rendered_depth=False,
                    enable_save_rendered_normals=False,
                    enable_save_rendered_alpha=False,
                    enable_save_rendered_extra_attrs=False,

                    #训练参数
                    extra_attrs_dim = 0,
                    eval_rate=1.0,
                    lr_args=LearningRate,
                    loss_weights=LossWeights,
          
                    init_degree=0,
                    max_sh_degree=3,
                    bg_color = [1, 1, 1],

                    iterations=30_000,
                    sh_increase_interval=1000,
                    save_iterations=[15_000, 30_000],
                    save_interval=10_000,
                    eval_interval=1000,
                    opacity_reset_interval=3000,
                    densify_from_iter=500,
                    densification_interval=100,
                    densify_until_iter=15_000,
                    densify_grad_threshold= 0.0002,

                 ):
        print(lr_args)
        print(loss_weights)
        gaussians = GaussianModel(init_degree,max_sh_degree,extra_attrs_dim)
        cams = None

        #载入COLMAP数据或者提供的相机参数
        if colmap_path is not None:
            print(f"Using COLMAP path: {colmap_path}")
            try:
                cameras_extrinsic_file = os.path.join(colmap_path, "sparse/0", "images.bin")
                cameras_intrinsic_file = os.path.join(colmap_path, "sparse/0", "cameras.bin")
                cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
                cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
            except:
                cameras_extrinsic_file = os.path.join(colmap_path, "sparse/0", "images.txt")
                cameras_intrinsic_file = os.path.join(colmap_path, "sparse/0", "cameras.txt")
                cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
                cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
            if images_folder is None:
                images_folder = os.path.join(colmap_path, "images/")
            cam_infos_unsorted = readColmapCameras(
                cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,images_folder=images_folder,depths_folder=depths_folder,
                normals_folder=normals_folder, alphas_folder=alphas_folder, extra_attrs_folder=extra_attrs_folder,Height=height, Width=width)
            cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
            nerf_normalization = getNerfppNorm(cam_infos)
            bin_path = os.path.join(colmap_path, "sparse/0/points3D.bin")
            txt_path = os.path.join(colmap_path, "sparse/0/points3D.txt")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            if rgb.max() > 1.0:
                rgb = rgb / 255.0
            gaussians.create_from_pcd(xyz, rgb,  nerf_normalization["radius"],add_skybox)
            cams = nn.ModuleList(cam_infos)
   
        elif w2c is not None and intrinsics is not None and (images is not None or (height is not None and width is not None)) and (xyz is not None or pretrained_path is not None):
            print("Using provided camera parameters.")
            cam_infos_unsorted = readCameras(cam_extrinsics=w2c, cam_intrinsics=intrinsics, images=images, depths=depths, normals=normals, alphas=alphas, extra_attrs=extra_attrs, Height=height, Width=width)
            cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
            nerf_normalization = getNerfppNorm(cam_infos)
            if xyz is not None:
                if rgb is None:
                    rgb = np.ones_like(xyz, dtype=np.float32)* 0.5
                if rgb.max() > 1.0:
                    rgb = rgb / 255.0
                gaussians.create_from_pcd(xyz, rgb, nerf_normalization["radius"],add_skybox)
            cams = nn.ModuleList(cam_infos)
        else:
            raise ValueError("Either colmap_path or w2c, intrinsics, images and (xyz or pretrained_path) must be provided.")
        
        #载入预训练模型
        if pretrained_path is not None:
            if pretrained_path.endswith(".ply"):
                print(f"Loading pretrained model from {pretrained_path}")
                gaussians.load_ply(pretrained_path)
            elif os.path.isdir(pretrained_path):
                print(f"Loading pretrained model from directory {pretrained_path}")
                gaussians.load_ply(os.path.join(pretrained_path, "model.ply"))
                cams.load_state_dict(torch.load(os.path.join(pretrained_path, "cameras.pth",),weights_only=True))
                gaussians._extra_attrs = nn.Parameter(
                    torch.load(os.path.join(pretrained_path, "extra_attrs.pth"),weights_only=True).cuda(), requires_grad=True)
            else:
                raise ValueError("Pretrained path must be a directory or a .ply file.")
            
        #设置训练集和验证集
        if eval_rate <1:
            self.indices_for_eval = [idx for idx in range(len(cams)) if idx % int(1/eval_rate) == 0]
        else:
            print("eval_rate is set to 1, all cameras will be used for evaluation.")
            self.indices_for_eval = [idx for idx in range(len(cams))]
        if not enable_train_all:
            self.indices_for_train = [idx for idx in range(len(cams)) if idx not in self.indices_for_eval]
        else:
            print("enable_train_all is set to True, all cameras will be used for training.")
            self.indices_for_train = [idx for idx in range(len(cams))]
        print(f"Number of cameras: {len(cams)}")
        print(f"Number of cameras for training: {len(self.indices_for_train)}, for evaluation: {len(self.indices_for_eval)}")

        #设置相机优化器
        if not enable_cam_update:
            cams.requires_grad_(False)
            print("enable_cam_update is set to False, cameras will not be updated during training.")
            self.optimizer_cam = None
        else:
            print("enable_cam_update is set to True, cameras will be updated during training.")
            self.optimizer_cam = torch.optim.Adam(cams.parameters(), lr=lr_args["cam_lr"])

        #print一些训练设置
        if not enable_densification:
            print("enable_densification is set to False, densification will not be performed during training.")
        if not enable_reset_opacity:
            print("enable_reset_opacity is set to False, opacity will not be reset during training.")
        print(f"Model will be saved to {save_dir}")
        #设置gs优化器
        gaussians.training_setup(lr_args)

        #初始化参数
        self.gaussians = gaussians
        self.cams = cams
        self.enable_densification = enable_densification
        self.enable_reset_opacity = enable_reset_opacity
        self.enable_save_rendered_images = enable_save_rendered_images
        self.enable_save_rendered_depth = enable_save_rendered_depth
        self.enable_save_rendered_normals = enable_save_rendered_normals
        self.enable_save_rendered_alpha = enable_save_rendered_alpha
        self.enable_save_rendered_extra_attrs= enable_save_rendered_extra_attrs
       
        self.densify_from_iter = densify_from_iter
        self.densification_interval = densification_interval
        self.opacity_reset_interval = opacity_reset_interval
        self.densify_grad_threshold = densify_grad_threshold
        self.densify_until_iter = densify_until_iter
        self.iterations = iterations
        self.save_iterations = save_iterations
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.eval_rate = eval_rate
        self.save_dir = save_dir
        self.sh_increase_interval = sh_increase_interval
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.loss_weights = loss_weights
        #初始化进度条
        pbar = Progress(TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            MofNCompleteColumn(),
            SpinnerColumn(),
            RenderableColumn())
        pbar.start()
        self.pbar = pbar
        self.train_task = self.pbar.add_task("[red]Training...", total= self.iterations)
        self.eval_task = self.pbar.add_task("[green]Evaluating...", total= len(self.indices_for_eval))
    def train(self,):
        indices_random = deepcopy(self.indices_for_train)
        random.shuffle(indices_random)
        for iteration in range(self.iterations):

            self.gaussians.update_learning_rate(iteration)
            if iteration % self.sh_increase_interval == 0:
                self.gaussians.oneupSHdegree()
            if len(indices_random)==0:
                indices_random = deepcopy(self.indices_for_train)
                random.shuffle(indices_random)

            viewpoint_cam = self.cams[indices_random.pop()]
            render_pkg = render(viewpoint_cam, self.gaussians, self.background)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            depth,normals,alpha, extra_attrs = render_pkg["depth"], render_pkg["normals"], render_pkg["alpha"], render_pkg["extra_attrs"]
            gt_image = viewpoint_cam.image_gt.cuda()
            Ll1 = l1_loss(image, gt_image)
            ssim_value = ssim(image, gt_image)

            loss = Ll1 * self.loss_weights["rgb_l1_weight"] + (1.0 - ssim_value) * self.loss_weights["rgb_ssim_weight"]
            if viewpoint_cam.depth_gt is not None:
                depth_gt = viewpoint_cam.depth_gt.cuda()
                depth_loss = torch.nn.functional.mse_loss(depth, depth_gt)
                loss += depth_loss* self.loss_weights["depth_weight"]
            if viewpoint_cam.normal_gt is not None:
                normal_gt = viewpoint_cam.normal_gt.cuda()
                normals_loss = torch.nn.functional.mse_loss(normals, normal_gt)
                loss += normals_loss * self.loss_weights["normals_weight"]
            if viewpoint_cam.alpha_gt is not None:
                alpha_gt = viewpoint_cam.alpha_gt.cuda()
                alpha_loss =((alpha_gt==0).float()*alpha).mean()
                loss += alpha_loss * self.loss_weights["alpha_weight"]
            if viewpoint_cam.extra_attrs_gt is not None:
                extra_attrs_gt = viewpoint_cam.extra_attrs_gt.cuda()
                extra_attrs_loss = torch.nn.functional.cross_entropy(extra_attrs[None], extra_attrs_gt[None])
                loss += extra_attrs_loss * self.loss_weights["extra_attrs_weight"]
            
            loss.backward()

            with torch.no_grad():
                if (iteration in self.save_iterations) or (iteration % self.save_interval == 0) or (iteration == self.iterations - 1):
                    os.makedirs(os.path.join(self.save_dir, f"{iteration}"), exist_ok=True)
                    save_path = os.path.join(self.save_dir,f"{iteration}", f"model.ply")
                    self.gaussians.save_ply(save_path)
                    save_path = os.path.join(self.save_dir,f"{iteration}", f"cameras.pth")
                    torch.save(self.cams.state_dict(), save_path)
                    save_path = os.path.join(self.save_dir,f"{iteration}", f"extra_attrs.pth")
                    torch.save(self.gaussians._extra_attrs.detach().cpu(), save_path)
                if iteration < self.densify_until_iter and self.enable_densification:
                    # Keep track of max radii in image-space for pruning
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    if iteration > self.densify_from_iter and iteration % self.densification_interval == 0:
                        size_threshold = 20 if iteration > self.opacity_reset_interval else None
                        self.gaussians.densify_and_prune(self.densify_grad_threshold, 0.005, self.gaussians.spatial_lr_scale, size_threshold, radii)

                    if iteration+1 % self.opacity_reset_interval == 0 and self.enable_reset_opacity:
                        self.gaussians.reset_opacity()

            self.gaussians.optimizer.step()
            self.gaussians.optimizer.zero_grad(set_to_none = True)
            if self.optimizer_cam is not None:
                self.optimizer_cam.step()
                self.optimizer_cam.zero_grad(set_to_none = True)
            self.pbar.update(self.train_task, advance=1,description=f"[red]Training...  | Loss: {loss.item():.4f}")

            if iteration % self.eval_interval == 0 or iteration == self.iterations - 1:
                self.eval(iteration)

    @torch.no_grad()
    def eval(self,iteration=0):
        self.pbar.reset(self.eval_task)
        l1_record = []
        ssim_record = []
        psnr_record = []
        for idx,ind_for_eval in enumerate(self.indices_for_eval):
            viewpoint_cam = self.cams[ind_for_eval]
            render_pkg = render(viewpoint_cam, self.gaussians, self.background)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            depth,normals,alpha, extra_attrs = render_pkg["depth"], render_pkg["normals"], render_pkg["alpha"], render_pkg["extra_attrs"]
            self.save_outputs(render_pkg,viewpoint_cam,iteration)
            gt_image = viewpoint_cam.image_gt.cuda()
            Ll1 = l1_loss(image, gt_image).item()
            ssim_value = ssim(image, gt_image).item()
            psnr_value = psnr(image, gt_image).mean().item()
            l1_record.append(Ll1)
            ssim_record.append(ssim_value)
            psnr_record.append(psnr_value)
            self.pbar.update(self.eval_task, advance=1,description=f"[green]Evaluating... {idx} | PSNR: {psnr_value:.4f} | SSIM: {ssim_value:.4f} | L1: {Ll1:.4f}")
        l1_mean = sum(l1_record) / len(l1_record)
        ssim_mean = sum(ssim_record) / len(ssim_record)
        psnr_mean = sum(psnr_record) / len(psnr_record) 
        
        print(f"Evaluation results of {iteration}: PSNR: {psnr_mean:.4f}, SSIM: {ssim_mean:.4f}, L1: {l1_mean:.4f}")
        return None
    def save_outputs(self,render_pkg,viewpoint_cam,iteration):
        rgb,depth,normals,alpha, extra_attrs =render_pkg["render"], render_pkg["depth"], render_pkg["normals"], render_pkg["alpha"], render_pkg["extra_attrs"]
        image_name = viewpoint_cam.image_name.split('.')[0]
        if self.enable_save_rendered_images:
            os.makedirs(os.path.join(self.save_dir, f"{iteration}",f"rendered_images"), exist_ok=True)
            save_path = os.path.join(self.save_dir,f"{iteration}",f"rendered_images", f"{image_name}.png")
            rgb = (rgb * 255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            rgb = Image.fromarray(rgb)
            rgb.save(save_path)
        if self.enable_save_rendered_depth:
            os.makedirs(os.path.join(self.save_dir, f"{iteration}",f"rendered_depth"), exist_ok=True)
            save_path = os.path.join(self.save_dir,f"{iteration}",f"rendered_depth", f"{image_name}.png")
            depth = depth.float().detach().cpu().numpy()
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)
            depth = (CMAP(depth)[:, :, :3])[:, :, ::-1]
            depth = Image.fromarray((depth * 255).astype(np.uint8))
            depth.save(save_path)
        if self.enable_save_rendered_normals:
            os.makedirs(os.path.join(self.save_dir, f"{iteration}",f"rendered_normals"), exist_ok=True)
            save_path = os.path.join(self.save_dir,f"{iteration}",f"rendered_normals", f"{image_name}.npy")
            normals = normals.float().detach().cpu().numpy()
            np.save(save_path, normals)
        if self.enable_save_rendered_alpha:
            os.makedirs(os.path.join(self.save_dir, f"{iteration}",f"rendered_alpha"), exist_ok=True)
            save_path = os.path.join(self.save_dir,f"{iteration}",f"rendered_alpha", f"{image_name}.npy")
            alpha = alpha.float().detach().cpu().numpy()
            np.save(save_path, alpha)
        if self.enable_save_rendered_extra_attrs:
            os.makedirs(os.path.join(self.save_dir, f"{iteration}",f"rendered_extra_attrs"), exist_ok=True)
            save_path = os.path.join(self.save_dir,f"{iteration}",f"rendered_extra_attrs", f"{image_name}.npy")
            extra_attrs = extra_attrs.float().detach().cpu().numpy()
            np.save(save_path, extra_attrs)
    def save_colmap(self,save_dir,save_image=False):
        intrinsics = []
        extrinsics = []
        if save_image:
            image_names = [cam.image_name for cam in self.cams]
        else:
            image_names = [cam.image_path for cam in self.cams]
        os.makedirs(os.path.join(save_dir, "sparse/0"), exist_ok=True)
        if save_image:
            os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
            print(f"Saving images to {os.path.join(save_dir, 'images')}")
        for cam in self.cams:
            intr = np.zeros((3, 3), dtype=np.float32)
            intr[0, 0] = cam.focal_length_x
            intr[1, 1] = cam.focal_length_y
            intr[0, 2] = int(cam.image_width/2)
            intr[1, 2] = int(cam.image_height/2)
            intrinsics.append(intr)
            extr = cam.world_view_transform.detach().cpu().transpose(0, 1).numpy()[:3, :4]
            extrinsics.append(extr)
            if save_image:
                image = (cam.image_gt * 255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
                image = Image.fromarray(image)
                image.save(os.path.join(save_dir, "images", f"{cam.image_name}"))
        intrinsics = np.stack(intrinsics, axis=0)
        extrinsics = np.stack(extrinsics, axis=0)
        print(f"Saving COLMAP data to {save_dir}")
        rec = batch_np_matrix_to_pycolmap_wo_track(
            points3d=self.gaussians._xyz.detach().cpu().numpy(),
            points_rgb=None,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            image_names=image_names,
            image_size=(self.cams[0].image_width, self.cams[0].image_height),
            shared_camera=False,
            camera_type="PINHOLE",
        )
        rec.write_text(os.path.join(save_dir, "sparse/0/"))
   
if __name__ == "__main__":
    # LearningRate["position_lr_init"]=0
    # LearningRate["position_lr_final"]=0
    # LearningRate["position_lr_delay_mult"]=0.
    trainer = GSTrainer(
        #要么提供COLMAP路径，要么提供相机参数
        colmap_path="/nvme0/public_data/Occupancy/proj/vis/gaussian-splatting/inputs/full/figurines",
        # pretrained_path="/nvme0/public_data/Occupancy/proj/vggt/examples/drive/900",
        save_dir="/nvme0/public_data/Occupancy/proj/vggt/examples/figurines",
        height=450,
        width=800,
        #训练和测试设置
        enable_densification=True,
        enable_reset_opacity=True,
        enable_train_all=True,
        enable_cam_update=False,
        enable_save_rendered_images=True,
        enable_save_rendered_depth=True,
        enable_save_rendered_normals=False,
        enable_save_rendered_alpha=False,
        enable_save_rendered_extra_attrs=False,

        #训练参数
        extra_attrs_dim=0,
        eval_rate=0.1,
        lr_args=LearningRate,
        loss_weights=LossWeights,
       
        init_degree=3,
        max_sh_degree=3,
        bg_color = [1, 1, 1],

        iterations=10_000,
        save_interval=1000,
        eval_interval=250)
    # trainer.save_colmap(trainer.save_dir, save_image=True)
    trainer.train()
    print("Training finished.")