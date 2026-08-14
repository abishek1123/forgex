#!/usr/bin/env python3
"""KLA image restoration -- inference.  THE SUBMISSION SCRIPT.

    python inference.py --input <test_images_dir> --output <restored_dir>

Reads every image in --input, restores it at 2x resolution, writes it to
--output under the same filename. Supports .npy (primary), .png, .tif/.tiff.
Falls back to CPU automatically. Requires only torch + numpy (+ Pillow, and
only if the inputs are image files rather than .npy).

Owner: Person D (Compute & packaging). THIS FILE MUST RUN AS-IS ON A MACHINE
THAT IS NOT YOURS. Test it from a fresh clone with CUDA_VISIBLE_DEVICES="".
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
from model import Restorer  # noqa: E402

EXTS = (".npy", ".png", ".tif", ".tiff")


def load_image(path):
    """-> (float32 HxW array roughly in [0,1], metadata for writing back)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path).astype(np.float32), ("npy", None)
    from PIL import Image
    im = Image.open(path)
    a = np.array(im)
    if a.ndim == 3:
        a = a.mean(axis=2)
    scale = 65535.0 if a.dtype == np.uint16 else 255.0
    return a.astype(np.float32) / scale, ("img", (ext, a.dtype, scale))


def save_image(path, arr, meta):
    kind, extra = meta
    if kind == "npy":
        np.save(path, arr.astype(np.float32))
        return
    from PIL import Image
    _, dtype, scale = extra
    out = np.clip(arr * scale, 0, scale).astype(dtype)
    Image.fromarray(out).save(path)


def restore_batch(model, xs, device, use_amp, tta):
    """xs: list of (H,W) float32, all same shape -> list of (2H,2W) float32 in [0,1]."""
    t = torch.from_numpy(np.stack(xs))[:, None].to(device)
    variants = [t]
    if tta:
        variants = [t, t.flip(-1), t.flip(-2), t.flip(-1, -2),
                    t.transpose(-1, -2), t.transpose(-1, -2).flip(-1),
                    t.transpose(-1, -2).flip(-2), t.transpose(-1, -2).flip(-1, -2)]
    outs = []
    with torch.inference_mode():
        for k, v in enumerate(variants):
            with torch.autocast("cuda", enabled=use_amp):
                o = model(v.contiguous()).float()
            if tta:
                if k == 1:
                    o = o.flip(-1)
                elif k == 2:
                    o = o.flip(-2)
                elif k == 3:
                    o = o.flip(-1, -2)
                elif k == 4:
                    o = o.transpose(-1, -2)
                elif k == 5:
                    o = o.flip(-1).transpose(-1, -2)
                elif k == 6:
                    o = o.flip(-2).transpose(-1, -2)
                elif k == 7:
                    o = o.flip(-1, -2).transpose(-1, -2)
            outs.append(o)
    out = torch.stack(outs).mean(0).clamp(0, 1)[:, 0].cpu().numpy()
    return [out[i] for i in range(out.shape[0])]


def main():
    p = argparse.ArgumentParser(description="Restore degraded semiconductor inspection images (2x).")
    p.add_argument("--input", required=True, help="directory of degraded images")
    p.add_argument("--output", required=True, help="directory to write restored images to")
    p.add_argument("--weights", default=os.path.join(HERE, "weights", "model.pt"))
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--fp16", action="store_true", default=True, help="half precision on CUDA (default on)")
    p.add_argument("--no-fp16", dest="fp16", action="store_false")
    p.add_argument("--tta", action="store_true", help="8x self-ensemble: ~+0.2 dB, 8x slower")
    p.add_argument("--batch", type=int, default=0,
                   help="images per forward pass; 0 = auto (16 on GPU, 1 on CPU). Batching "
                        "amortises kernel-launch overhead on GPU but hurts CPU cache locality.")
    a = p.parse_args()

    device = torch.device("cuda" if (a.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu")
    use_amp = a.fp16 and device.type == "cuda"
    if a.batch <= 0:
        a.batch = 16 if device.type == "cuda" else 1

    if not os.path.isfile(a.weights):
        sys.exit(f"ERROR: weights not found at {a.weights}\n"
                 f"Pass --weights, or see the README for the download link.")
    ck = torch.load(a.weights, map_location="cpu")
    model = Restorer(**ck.get("config", {})).to(device).eval()
    model.load_state_dict(ck["state_dict"])
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    files = sorted(f for f in os.listdir(a.input) if f.lower().endswith(EXTS))
    if not files:
        sys.exit(f"ERROR: no {'/'.join(EXTS)} files found in {a.input}")
    os.makedirs(a.output, exist_ok=True)
    print(f"device={device}  fp16={use_amp}  tta={a.tta}  files={len(files)}")

    # Warm-up so the first image does not absorb CUDA init time.
    first, _ = load_image(os.path.join(a.input, files[0]))
    restore_batch(model, [first], device, use_amp, False)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    # Batch images of identical size together: one forward per batch instead of
    # per image. Mixed sizes (128x128 and 256x256 test inputs) group separately.
    pending = {}   # shape -> list of (name, array, meta)
    def flush(shape):
        group = pending.pop(shape, [])
        for i in range(0, len(group), a.batch):
            chunk = group[i:i + a.batch]
            ys = restore_batch(model, [g[1] for g in chunk], device, use_amp, a.tta)
            for (name, _, meta), y in zip(chunk, ys):
                save_image(os.path.join(a.output, name), y, meta)
    for name in files:
        x, meta = load_image(os.path.join(a.input, name))
        pending.setdefault(x.shape, []).append((name, x, meta))
        if len(pending[x.shape]) >= a.batch:
            flush(x.shape)
    for shape in list(pending):
        flush(shape)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    print(f"restored {len(files)} images in {dt:.2f}s  ({dt/len(files)*1000:.2f} ms/image)")
    print(f"written to {a.output}")


if __name__ == "__main__":
    main()
