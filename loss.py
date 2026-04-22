import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp


class ImageGradientLoss(nn.Module):
    """
    Penalizes blurry outputs by matching image gradients (edges) between pred and target.
    The mean of two frames smears edges; this loss directly punishes that.
    """
    def __init__(self):
        super(ImageGradientLoss, self).__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def forward(self, pred, target):
        B, C, H, W = pred.shape
        pred_flat = pred.view(B * C, 1, H, W)
        target_flat = target.view(B * C, 1, H, W)

        pred_gx = F.conv2d(pred_flat, self.sobel_x, padding=1)
        pred_gy = F.conv2d(pred_flat, self.sobel_y, padding=1)
        target_gx = F.conv2d(target_flat, self.sobel_x, padding=1)
        target_gy = F.conv2d(target_flat, self.sobel_y, padding=1)

        return (torch.abs(pred_gx - target_gx) + torch.abs(pred_gy - target_gy)).mean()

def gaussian_window(window_size, sigma):
    """Generates a 1D Gaussian window."""
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    """Generates a 2D Gaussian window for SSIM."""
    _1D_window = gaussian_window(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

class SSIMLoss(nn.Module):
    """
    Calculates the Structural Similarity Index Measure (SSIM) Loss.
    Since SSIM is 1 for perfect identical images, we return (1 - SSIM) to create a minimizable loss.
    """
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device)
            self.window = window
            self.channel = channel

        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return 1.0 - ssim_map.mean()
        else:
            return 1.0 - ssim_map.mean(1).mean(1).mean(1)


class CharbonnierLoss(nn.Module):
    """
    A smooth approximation of L1 loss. 
    It is more robust to outliers and prevents vanishing gradients near zero, 
    making it standard for video frame interpolation tasks.
    """
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        # rho(x) = sqrt(x^2 + eps^2)
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return loss.mean()


class PerceptualLoss(nn.Module):
    """
    VGG16 perceptual loss up to relu3_3 (feature index 16).
    Operates in a non-linear feature space where averaged frames are distinct from real
    frames, breaking the degenerate minimum that pixel-wise losses all share.
    """
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        import torchvision.models as models
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:17])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, pred, target):
        pred   = (pred.clamp(0.0, 1.0)   - self.mean) / self.std
        target = (target.clamp(0.0, 1.0) - self.mean) / self.std
        return F.l1_loss(self.feature_extractor(pred), self.feature_extractor(target))


class FlowConsistencyLoss(nn.Module):
    """
    Forward-backward consistency: flow_0 + warp(flow_1, flow_0) ≈ 0.
    Following flow_0 from frame_0 to the intermediate frame, then flow_1 back,
    should return to the origin. Gradient is texture-independent.
    """
    def __init__(self):
        super(FlowConsistencyLoss, self).__init__()
        self.charb = CharbonnierLoss()

    def _warp(self, flow_to_warp, flow_coords):
        B, _, H, W = flow_coords.shape
        xs = torch.arange(W, device=flow_coords.device, dtype=torch.float32)
        ys = torch.arange(H, device=flow_coords.device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        base = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        sampling = base + flow_coords
        norm_x = 2.0 * sampling[:, 0] / (W - 1) - 1.0
        norm_y = 2.0 * sampling[:, 1] / (H - 1) - 1.0
        grid = torch.stack([norm_x, norm_y], dim=-1)
        return F.grid_sample(flow_to_warp, grid, mode='bilinear',
                             padding_mode='border', align_corners=True)

    def forward(self, flow_0, flow_1):
        warped_flow_1 = self._warp(flow_1, flow_0)
        return self.charb(flow_0 + warped_flow_1, torch.zeros_like(flow_0))


class FlowTVLoss(nn.Module):
    """
    Total variation on both flow fields. Encourages spatial smoothness,
    propagating flow estimates from textured edges into smooth interiors.
    """
    def forward(self, flow_0, flow_1):
        def tv(f):
            return (torch.abs(f[:, :, 1:, :] - f[:, :, :-1, :]).mean() +
                    torch.abs(f[:, :, :, 1:] - f[:, :, :, :-1]).mean())
        return tv(flow_0) + tv(flow_1)


class FlowSupervisionLoss(nn.Module):
    """
    Charbonnier loss between predicted flows and RAFT pseudo-GT flows.
    Direct flow supervision bypasses the bilinear-sampling gradient dead zone
    in textureless image regions.
    """
    def __init__(self):
        super(FlowSupervisionLoss, self).__init__()
        self.charb = CharbonnierLoss()

    def forward(self, flow_0, flow_1, gt_flow_0, gt_flow_1):
        return self.charb(flow_0, gt_flow_0) + self.charb(flow_1, gt_flow_1)


class VFICombinedLoss(nn.Module):
    """
    Combines SSIM, Charbonnier, gradient, warp supervision, and L1 weight regularization.

    Warp supervision (warped_0, warped_1 vs target) is the primary fix for ghosting: it
    forces each flow to independently match the target before blending, so the model cannot
    collapse to a trivial 0.5/0.5 average of the input frames.
    """
    def __init__(self, ssim_weight=0.1, charbonnier_weight=0.1,
                 gradient_weight=0.1, warp_weight=0.6, l1_weight=1e-4,
                 perceptual_weight=0.05, flow_supervision_weight=0.01,
                 flow_consistency_weight=0.01, flow_tv_weight=0.005):
        super(VFICombinedLoss, self).__init__()
        self.ssim_loss             = SSIMLoss()
        self.charbonnier_loss      = CharbonnierLoss()
        self.gradient_loss         = ImageGradientLoss()
        self.perceptual_loss       = PerceptualLoss()
        self.flow_supervision_loss = FlowSupervisionLoss()
        self.flow_consistency_loss = FlowConsistencyLoss()
        self.flow_tv_loss          = FlowTVLoss()

        self.ssim_weight              = ssim_weight
        self.charbonnier_weight       = charbonnier_weight
        self.gradient_weight          = gradient_weight
        self.warp_weight              = warp_weight
        self.l1_weight                = l1_weight
        self.perceptual_weight        = perceptual_weight
        self.flow_supervision_weight  = flow_supervision_weight
        self.flow_consistency_weight  = flow_consistency_weight
        self.flow_tv_weight           = flow_tv_weight

    def forward(self, pred_frame, target_frame, model_parameters,
                warped_0=None, warped_1=None,
                flow_0=None, flow_1=None,
                gt_flow_0=None, gt_flow_1=None):
        # 1. Reconstruction losses on the final blended output
        loss_ssim   = self.ssim_loss(pred_frame, target_frame)
        loss_charb  = self.charbonnier_loss(pred_frame, target_frame)
        loss_grad   = self.gradient_loss(pred_frame, target_frame)
        loss_percep = self.perceptual_loss(pred_frame, target_frame)

        # 2. Warp supervision
        loss_warp = torch.tensor(0., device=pred_frame.device)
        if warped_0 is not None and warped_1 is not None:
            loss_warp = (self.charbonnier_loss(warped_0, target_frame) +
                         self.charbonnier_loss(warped_1, target_frame))

        # 3. RAFT pseudo-GT flow supervision (training only; skipped when gt_flow_* is None)
        loss_flow_sup = torch.tensor(0., device=pred_frame.device)
        if flow_0 is not None and gt_flow_0 is not None:
            loss_flow_sup = self.flow_supervision_loss(flow_0, flow_1, gt_flow_0, gt_flow_1)

        # 4. Forward-backward flow consistency (texture-independent gradient signal)
        loss_flow_cons = torch.tensor(0., device=pred_frame.device)
        if flow_0 is not None and flow_1 is not None:
            loss_flow_cons = self.flow_consistency_loss(flow_0, flow_1)

        # 5. Flow total variation (spatial smoothness)
        loss_flow_tv = torch.tensor(0., device=pred_frame.device)
        if flow_0 is not None and flow_1 is not None:
            loss_flow_tv = self.flow_tv_loss(flow_0, flow_1)

        # 6. L1 weight regularization (for future pruning)
        l1_reg = torch.tensor(0., device=pred_frame.device)
        for param in model_parameters:
            l1_reg += torch.norm(param, p=1)

        total_loss = (self.ssim_weight             * loss_ssim      +
                      self.charbonnier_weight       * loss_charb     +
                      self.gradient_weight          * loss_grad      +
                      self.perceptual_weight        * loss_percep    +
                      self.warp_weight              * loss_warp      +
                      self.flow_supervision_weight  * loss_flow_sup  +
                      self.flow_consistency_weight  * loss_flow_cons +
                      self.flow_tv_weight           * loss_flow_tv   +
                      self.l1_weight                * l1_reg)

        return total_loss, loss_ssim, loss_charb, l1_reg
