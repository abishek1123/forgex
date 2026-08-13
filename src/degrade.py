"""Degradation model for the KLA image-restoration challenge.

Reverse-engineered from the supplied train/GT + train/NoisyLR pairs. KLA's
briefing deck states the pipeline is "speckle noise, down-sampling, additive
gaussian noise -- do not read into the order of it", so we randomise the order.

Fitting sigma(I)^2 = sigma_add^2 + sigma_mul^2 * I^2 on real pairs gives:
    sigma_mul  ~ U(0.13, 0.21)      (speckle: noise grows with brightness)
    sigma_add  ~ U(0.00, 0.07)      (plain additive gaussian)
and downsampling is an exact 2x2 box average (no extra blur kernel: fitting a
gaussian blur before decimation selected sigma = 0.0 on every pair tested).

We TRAIN on deliberately wider ranges than we measured, because the hidden test
set is explicitly out-of-distribution. See tools/calibrate.py for the evidence.
"""
import numpy as np
from scipy.ndimage import gaussian_filter

# What we actually measured on KLA's data.
OBSERVED = dict(speckle=(0.13, 0.21), gauss=(0.00, 0.07), blur_p=0.0, blur_sigma=(0.0, 0.0))

# What we train on: wider, plus occasional mild blur to cover the "soft and
# hazy" degradation described in the problem statement.
TRAIN = dict(speckle=(0.02, 0.30), gauss=(0.00, 0.18), blur_p=0.25, blur_sigma=(0.3, 0.9))


def box_down(x, f=2):
    """Exact f x f box average -- this is how KLA downsamples."""
    H, W = x.shape
    H, W = (H // f) * f, (W // f) * f
    return x[:H, :W].reshape(H // f, f, W // f, f).mean(axis=(1, 3))


def degrade(gt, rng, cfg=TRAIN):
    """Clean HR (H,W) float32 in [0,1]  ->  noisy LR (H/2,W/2) float32.

    The output may fall outside [0,1]; that is correct and matches KLA's data.
    `rng` must be a numpy Generator so every worker/epoch gets fresh noise.
    """
    x = np.asarray(gt, dtype=np.float32)
    s = float(rng.uniform(*cfg["speckle"]))
    g = float(rng.uniform(*cfg["gauss"]))
    ops = ["speckle", "down", "gauss"]
    rng.shuffle(ops)
    for op in ops:
        if op == "speckle":
            x = x * (1.0 + s * rng.standard_normal(x.shape).astype(np.float32))
        elif op == "gauss":
            x = x + g * rng.standard_normal(x.shape).astype(np.float32)
        else:
            if cfg.get("blur_p", 0.0) > 0 and rng.random() < cfg["blur_p"]:
                x = gaussian_filter(x, float(rng.uniform(*cfg["blur_sigma"])), mode="reflect")
            x = box_down(x)
    return np.ascontiguousarray(x, dtype=np.float32)
