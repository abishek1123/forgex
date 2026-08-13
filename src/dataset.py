"""Datasets. Owner: Person A (Data).

RestoreDataset mixes KLA's real NoisyLR files with pairs we generate ourselves
via degrade(). p_real controls the blend. Synthetic pairs are re-rolled every
time an image is drawn, so there is no fixed noise pattern to memorise.
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

from degrade import degrade, TRAIN


def list_ids(d):
    return sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(d, "*.npy")))


def make_split(gt_dir, n_val=200, seed=0):
    """Deterministic train/val split. Everyone must use the same seed."""
    ids = list_ids(gt_dir)
    perm = np.random.default_rng(seed).permutation(len(ids))
    val = {ids[i] for i in perm[:n_val]}
    return [i for i in ids if i not in val], sorted(val)


def _dihedral(a, k):
    if k & 1:
        a = a[:, ::-1]
    if k & 2:
        a = a[::-1, :]
    if k & 4:
        a = a.T
    return np.ascontiguousarray(a)


class RestoreDataset(Dataset):
    """Yields (lr, hr) float32 tensors: (1, crop, crop) and (1, 2*crop, 2*crop)."""

    def __init__(self, gt_dir, lr_dir, ids, crop=64, p_real=0.3, cfg=TRAIN, seed=0, length=None):
        self.gt_dir, self.lr_dir = gt_dir, lr_dir
        self.ids = list(ids)
        self.crop, self.p_real, self.cfg, self.seed = crop, p_real, cfg, seed
        self.length = int(length) if length else len(self.ids)

    def __len__(self):
        return self.length

    def _rng(self):
        if not hasattr(self, "_g"):
            info = torch.utils.data.get_worker_info()
            wid = 0 if info is None else info.id
            self._g = np.random.default_rng([self.seed, wid, os.getpid()])
        return self._g

    def __getitem__(self, idx):
        g = self._rng()
        name = self.ids[idx % len(self.ids)]
        gt = np.load(os.path.join(self.gt_dir, name + ".npy")).astype(np.float32)
        c = self.crop
        H, W = gt.shape
        y = int(g.integers(0, H // 2 - c + 1))
        x = int(g.integers(0, W // 2 - c + 1))
        hr = gt[2 * y:2 * (y + c), 2 * x:2 * (x + c)]

        if self.lr_dir is not None and g.random() < self.p_real:
            lr = np.load(os.path.join(self.lr_dir, name + ".npy")).astype(np.float32)[y:y + c, x:x + c]
        else:
            lr = degrade(hr, g, self.cfg)

        k = int(g.integers(8))
        lr, hr = _dihedral(lr, k), _dihedral(hr, k)
        return torch.from_numpy(lr)[None], torch.from_numpy(hr)[None]


class ValDataset(Dataset):
    """Full images, KLA's REAL NoisyLR only -- never synthetic. This is the
    number we trust, because it is measured on the actual degradation."""

    def __init__(self, gt_dir, lr_dir, ids):
        self.gt_dir, self.lr_dir, self.ids = gt_dir, lr_dir, list(ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        name = self.ids[i]
        lr = np.load(os.path.join(self.lr_dir, name + ".npy")).astype(np.float32)
        gt = np.load(os.path.join(self.gt_dir, name + ".npy")).astype(np.float32)
        return torch.from_numpy(lr)[None], torch.from_numpy(gt)[None], name
