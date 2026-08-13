"""Loss functions. Owner: Person C (Loss).

Default = Charbonnier + SSIM + gradient. Rationale:
  * Charbonnier (smooth L1) beats MSE on PSNR in practice.
  * SSIM is one of the three metrics KLA scores, so we optimise it directly.
  * The gradient term sharpens edges WHERE THE GROUND TRUTH HAS EDGES. Unlike a
    perceptual/GAN loss it has no mechanism to invent texture -- which matters,
    because a hallucinated texture on an inspection image is a fabricated defect.

Reference: Zhao et al., "Loss Functions for Image Restoration with Neural
Networks", IEEE Trans. Computational Imaging, 2017.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def charbonnier(x, y, eps=1e-3):
    return torch.sqrt((x.float() - y.float()) ** 2 + eps ** 2).mean()


def _gauss_window(ws, sigma, device, dtype):
    g = torch.arange(ws, dtype=torch.float32, device=device) - (ws - 1) / 2.0
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None].to(dtype)


def ssim(x, y, ws=11, sigma=1.5, C1=0.01 ** 2, C2=0.03 ** 2):
    """Mean SSIM over the batch. Inputs (B,1,H,W), data range 1.0.

    ALWAYS computed in float32. In fp16 the local variances go slightly
    negative and the denominator product underflows below 6e-5 (fp16's
    smallest normal), which yields Inf and then NaN gradients -- and NaN is
    scale-invariant, so GradScaler skips every step forever and the model
    never trains. Do not remove the .float() calls.
    """
    # autocast MUST be disabled explicitly: F.conv2d inside an autocast region
    # re-casts its inputs to fp16 no matter what .float() we call beforehand.
    with torch.amp.autocast(device_type=x.device.type, enabled=False):
        x, y = x.float(), y.float()
        w = _gauss_window(ws, sigma, x.device, x.dtype)
        pad = ws // 2
        mx = F.conv2d(x, w, padding=pad)
        my = F.conv2d(y, w, padding=pad)
        vx = (F.conv2d(x * x, w, padding=pad) - mx * mx).clamp_min(0)
        vy = (F.conv2d(y * y, w, padding=pad) - my * my).clamp_min(0)
        vxy = F.conv2d(x * y, w, padding=pad) - mx * my
        s = ((2 * mx * my + C1) * (2 * vxy + C2)) / ((mx * mx + my * my + C1) * (vx + vy + C2))
        return s.mean()


def gradient_loss(x, y, eps=1e-3):
    dx = lambda t: t[..., :, 1:] - t[..., :, :-1]
    dy = lambda t: t[..., 1:, :] - t[..., :-1, :]
    return charbonnier(dx(x), dx(y), eps) + charbonnier(dy(x), dy(y), eps)


class RestorationLoss(nn.Module):
    """mode='charbonnier' is the control run; mode='combo' is the default."""

    def __init__(self, mode="combo", w_ssim=0.15, w_grad=0.05):
        super().__init__()
        self.mode, self.w_ssim, self.w_grad = mode, w_ssim, w_grad

    def forward(self, pred, gt):
        parts = {"charb": charbonnier(pred, gt)}
        total = parts["charb"]
        if self.mode == "combo":
            parts["ssim"] = 1.0 - ssim(pred.clamp(0, 1), gt)
            parts["grad"] = gradient_loss(pred, gt)
            total = total + self.w_ssim * parts["ssim"] + self.w_grad * parts["grad"]
        return total, {k: float(v.detach()) for k, v in parts.items()}
