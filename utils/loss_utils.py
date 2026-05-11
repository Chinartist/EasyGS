import cv2
import numpy as np
import torch
import torch.nn.functional as F
from math import exp
from torch.autograd import Variable

try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
    _HAS_FUSED_SSIM = True
except ImportError:
    _HAS_FUSED_SSIM = False

C1 = 0.01 ** 2
C2 = 0.03 ** 2


class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, c1, c2, img1, img2):
        ssim_map = fusedssim(c1, c2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = c1
        ctx.C2 = c2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        grad = fusedssim_backward(ctx.C1, ctx.C2, img1, img2, opt_grad)
        return None, None, grad, None


def l1_loss(network_output, gt):
    return torch.abs(network_output - gt).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    window_1d = gaussian(window_size, 1.5).unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
    return Variable(window_2d.expand(channel, 1, window_size, window_size).contiguous())


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

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    if size_average:
        return ssim_map.mean()
    return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    if not _HAS_FUSED_SSIM:
        return ssim(img1, img2)
    return FusedSSIMMap.apply(C1, C2, img1, img2).mean()


def extract_and_match_features(img1: torch.Tensor, img2: torch.Tensor):
    if not hasattr(cv2, "SIFT_create"):
        return None

    img1_np = img1.permute(1, 2, 0).detach().cpu().numpy()
    img2_np = img2.permute(1, 2, 0).detach().cpu().numpy()
    img1_gray = cv2.cvtColor((img1_np * 255).astype("uint8"), cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor((img2_np * 255).astype("uint8"), cv2.COLOR_RGB2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)
    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return None

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    raw_matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for pair in raw_matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) <= 4:
        return None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
    _, mask = cv2.findFundamentalMat(
        src_pts,
        dst_pts,
        cv2.FM_RANSAC,
        ransacReprojThreshold=3.0,
        confidence=0.99,
    )
    if mask is None:
        return None

    mask = mask.flatten().astype(bool)
    inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i]]
    if not inlier_matches:
        return None
    return kp1, kp2, inlier_matches


def reprojection_loss(img1: torch.Tensor, img2: torch.Tensor, F=None, H=None, K=None, device="cuda"):
    match_res = extract_and_match_features(img1, img2)
    if match_res is None:
        return torch.tensor(0.0, device=device)

    kp1, kp2, matches = match_res
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    pts1_t = torch.tensor(pts1, dtype=torch.float32, device=device)
    pts2_t = torch.tensor(pts2, dtype=torch.float32, device=device)
    ones = torch.ones((pts1_t.shape[0], 1), device=device)
    pts1_h = torch.cat([pts1_t, ones], dim=1)
    pts2_h = torch.cat([pts2_t, ones], dim=1)

    if F is not None:
        F_t = torch.tensor(F, dtype=torch.float32, device=device)
        term1 = torch.matmul(pts2_h, F_t)
        term2 = torch.sum(term1 * pts1_h, dim=1)
        term3 = torch.matmul(F_t, pts1_h.transpose(0, 1))
        term4 = torch.sum(pts2_h * term3.transpose(0, 1), dim=1)
        return torch.mean(0.5 * (term2 ** 2 + term4 ** 2))

    if H is not None:
        H_t = torch.tensor(H, dtype=torch.float32, device=device)
        pts1_warped = torch.matmul(H_t, pts1_h.transpose(0, 1)).transpose(0, 1)
        pts1_warped = pts1_warped / (pts1_warped[:, 2:] + 1e-8)
        return torch.norm(pts1_warped[:, :2] - pts2_t, dim=1).mean()

    if K is not None and len(matches) >= 5:
        _, e_mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if e_mask is None:
            return torch.tensor(0.0, device=device)
        e_mask = e_mask.flatten().astype(bool)
        pts1_inliers = pts1[e_mask]
        pts2_inliers = pts2[e_mask]
        if len(pts1_inliers) < 5:
            return torch.tensor(0.0, device=device)

        _, rotation, translation, _ = cv2.recoverPose(pts1_inliers, pts2_inliers, K, distanceThresh=100)
        t = translation.reshape(3)
        t_x = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
        essential = t_x @ rotation
        essential_t = torch.tensor(essential, dtype=torch.float32, device=device)
        term1 = torch.matmul(pts2_h, essential_t)
        errors = torch.sum(term1 * pts1_h, dim=1) ** 2
        return errors.mean()

    return torch.norm(pts1_t - pts2_t, dim=1).mean()
