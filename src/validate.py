#!/usr/bin/env python3
"""Score a checkpoint on the held-out split. Owner: Person C.

    python src/validate.py --data data/train --ckpt runs/v1/best.pt

Prints PSNR / SSIM / LPIPS and milliseconds per image, and appends a row to
results.csv. That csv becomes slide 6 -- run it after every training run,
including the bad ones.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch


def _nm(v):
    v = [x for x in v if x == x]
    return float(np.mean(v)) if v else float('nan')
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import ValDataset, make_split      # noqa: E402
from metrics import lpips, psnr, ssim           # noqa: E402
from model import Restorer                      # noqa: E402


def _tta(model, t):
    """8-fold dihedral self-ensemble. Mirrors inference.py exactly."""
    vs = [t, t.flip(-1), t.flip(-2), t.flip(-1, -2), t.transpose(-1, -2),
          t.transpose(-1, -2).flip(-1), t.transpose(-1, -2).flip(-2),
          t.transpose(-1, -2).flip(-1, -2)]
    inv = [lambda o: o, lambda o: o.flip(-1), lambda o: o.flip(-2), lambda o: o.flip(-1, -2),
           lambda o: o.transpose(-1, -2), lambda o: o.flip(-1).transpose(-1, -2),
           lambda o: o.flip(-2).transpose(-1, -2), lambda o: o.flip(-1, -2).transpose(-1, -2)]
    return torch.stack([inv[k](model(v)) for k, v in enumerate(vs)]).mean(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/train")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n-val", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--csv", default="results.csv")
    p.add_argument("--tag", default="")
    p.add_argument("--baseline", action="store_true", help="also score plain bicubic")
    p.add_argument("--tta", action="store_true",
                   help="8x dihedral self-ensemble, same transform set as inference.py --tta")
    a = p.parse_args()

    device = torch.device(a.device)
    gt_dir, lr_dir = os.path.join(a.data, "GT"), os.path.join(a.data, "NoisyLR")
    _, val_ids = make_split(gt_dir, n_val=a.n_val, seed=a.seed)
    dl = DataLoader(ValDataset(gt_dir, lr_dir, val_ids), batch_size=1, num_workers=0)

    ck = torch.load(a.ckpt, map_location=device)
    model = Restorer(**ck["config"]).to(device).eval()
    model.load_state_dict(ck["state_dict"])

    rows = {"model": [[], [], [], []]}
    if a.baseline:
        rows["bicubic"] = [[], [], [], []]

    with torch.no_grad():
        for lr, gt, _ in dl:
            lr, gt = lr.to(device), gt.to(device)
            t0 = time.perf_counter()
            out = (_tta(model, lr) if a.tta else model(lr)).clamp(0, 1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000
            for arr, pred in [("model", out)] + ([("bicubic", torch.nn.functional.interpolate(
                    lr, scale_factor=2, mode="bicubic", align_corners=False).clamp(0, 1))] if a.baseline else []):
                r = rows[arr]
                r[0].append(psnr(pred, gt))
                r[1].append(ssim(pred, gt))
                r[2].append(lpips(pred, gt, device=device))
                r[3].append(dt if arr == "model" else 0.0)

    print(f"\n{'what':10} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'ms/img':>8}")
    for k, r in rows.items():
        print(f"{k:10} {np.mean(r[0]):8.3f} {np.mean(r[1]):8.4f} {_nm(r[2]):8.4f} {np.mean(r[3]):8.2f}")

    new = not os.path.isfile(a.csv)
    with open(a.csv, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["tag", "ckpt", "epoch", "psnr", "ssim", "lpips", "ms_per_img", "params_M"])
        r = rows["model"]
        w.writerow([(a.tag or os.path.basename(os.path.dirname(a.ckpt))) + ("+tta" if a.tta else ""),
                    a.ckpt, ck.get("epoch", -1),
                    round(float(np.mean(r[0])), 4), round(float(np.mean(r[1])), 5),
                    round(float(_nm(r[2])), 5), round(float(np.mean(r[3])), 3),
                    round(model.n_params() / 1e6, 3)])
    print(f"\nappended to {a.csv}")


if __name__ == "__main__":
    main()
