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
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass

C1 = 0.01 ** 2
C2 = 0.03 ** 2

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()


def extract_and_match_features(img1: torch.Tensor, img2: torch.Tensor):
    # 1. 读取图像
    img1 = img1.permute(1, 2, 0).detach().cpu().numpy()
    img1 = cv2.cvtColor((img1 * 255).astype("uint8"), cv2.COLOR_RGB2GRAY)  # 转换为灰度图
    img2 = img2.permute(1, 2, 0).detach().cpu().numpy()
    img2 = cv2.cvtColor((img2 * 255).astype("uint8"), cv2.COLOR_RGB2GRAY)  # 转换为灰度图

    # 2. 创建SIFT特征检测器
    sift = cv2.SIFT_create()

    # 3. 检测关键点并计算描述符
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 4. 使用FLANN进行特征匹配
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # 或使用空字典{}表示默认参数

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # 5. 应用Lowe's ratio test过滤匹配
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:  # 典型的ratio阈值
            good_matches.append(m)

    # 6. 如果有足够的匹配点，进行RANSAC几何验证
    if len(good_matches) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

        # 使用RANSAC估计基础矩阵
        _, mask = cv2.findFundamentalMat(
            src_pts, dst_pts,
            cv2.FM_RANSAC,
            ransacReprojThreshold=3.0,
            confidence=0.99
        )

        # 应用掩码获取内点
        mask = mask.flatten().astype(bool)
        inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i]]

        return kp1, kp2, inlier_matches
    else:
        print("匹配点不足，无法进行RANSAC验证")
        return None


def reprojection_loss(img1: torch.Tensor, img2: torch.Tensor, F=None, H=None, K=None, device='cuda'):
    match_res = extract_and_match_features(img1, img2)
    if match_res is None:
        return torch.tensor(0.0, device=device, requires_grad=True)

    kp1, kp2, matches = match_res
    if len(matches) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # 将匹配点转换为坐标数组
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # 转换为PyTorch张量
    pts1_t = torch.tensor(pts1, dtype=torch.float32, device=device, requires_grad=False)
    pts2_t = torch.tensor(pts2, dtype=torch.float32, device=device, requires_grad=False)

    # 添加齐次坐标
    ones = torch.ones((pts1_t.shape[0], 1), device=device)
    pts1_h = torch.cat([pts1_t, ones], dim=1)  # [N, 3]
    pts2_h = torch.cat([pts2_t, ones], dim=1)  # [N, 3]

    if F is not None:
        # 基础矩阵F的重投影误差: x2^T * F * x1
        F_t = torch.tensor(F, dtype=torch.float32, device=device)

        # 计算 x2^T * F
        term1 = torch.matmul(pts2_h, F_t)  # [N, 3]

        # 计算 (x2^T * F) * x1
        term2 = torch.sum(term1 * pts1_h, dim=1)  # [N]

        # 对称重投影误差
        term3 = torch.matmul(F_t, pts1_h.transpose(0, 1))  # [3, N]
        term4 = torch.sum(pts2_h * term3.transpose(0, 1), dim=1)  # [N]

        # 总误差
        errors = 0.5 * (term2 ** 2 + term4 ** 2)
        loss = torch.mean(errors)

    elif H is not None:
        # 单应矩阵H的重投影误差: ||x2 - H*x1||
        H_t = torch.tensor(H, dtype=torch.float32, device=device)

        # 计算 H*x1
        pts1_warped = torch.matmul(H_t, pts1_h.transpose(0, 1)).transpose(0, 1)  # [N, 3]

        # 归一化齐次坐标
        pts1_warped = pts1_warped / (pts1_warped[:, 2:] + 1e-8)

        # 计算欧氏距离
        errors = torch.norm(pts1_warped[:, :2] - pts2_t, dim=1)
        loss = torch.mean(errors)

    elif K is not None and len(matches) >= 5:
        # 如果有相机内参，可以计算本质矩阵并得到重投影误差
        # 首先计算本质矩阵
        _, E_mask = cv2.findEssentialMat(pts1, pts2, K,
                                         method=cv2.RANSAC,
                                         prob=0.999,
                                         threshold=1.0)
        E_mask = E_mask.flatten().astype(bool)

        # 只保留内点
        pts1_inliers = pts1[E_mask]
        pts2_inliers = pts2[E_mask]

        if len(pts1_inliers) >= 5:
            _, R, t, _ = cv2.recoverPose(
                pts1_inliers, pts2_inliers, K,
                distanceThresh=100
            )

            # 构建本质矩阵
            t_x = np.array([[0, -t[2], t[1]],
                            [t[2], 0, -t[0]],
                            [-t[1], t[0], 0]])
            E = t_x @ R

            # 计算重投影误差
            E_t = torch.tensor(E, dtype=torch.float32, device=device)
            term1 = torch.matmul(pts2_h, E_t)
            term2 = torch.sum(term1 * pts1_h, dim=1)
            errors = term2 ** 2
            loss = torch.mean(errors)
        else:
            loss = torch.tensor(0.0, device=device)
    else:
        # 默认使用欧氏距离作为loss
        errors = torch.norm(pts1_t - pts2_t, dim=1)
        loss = torch.mean(errors)

    return loss
