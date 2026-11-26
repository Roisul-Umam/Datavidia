"""Colour-jitter covariates per image (used to deconfound zero-shot scores).

columns: share of saturated purple/magenta pixels (hue 260-340 deg), share of saturated blue
pixels (hue 190-260), and whether blue is the brightest channel on average.
writes exp/cache/colour_covariates.npz with arrays pre, post of shape (N, 3)
"""
import sys
from multiprocessing.dummy import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import CACHE  # noqa: E402

raw = {w: np.load(CACHE / f"raw512_{w}.npy", mmap_mode="r") for w in ("pre", "post")}


def feats(args):
    w, k = args
    x = np.asarray(raw[w][k][::4, ::4])
    hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = hsv[..., 0] * 2, hsv[..., 1] / 255, hsv[..., 2] / 255
    sat = (s > 0.15) & (v > 0.1)
    m = x.reshape(-1, 3).astype(float).mean(0)
    return w, k, (sat & (h >= 260) & (h <= 340)).mean(), (sat & (h >= 190) & (h < 260)).mean(), float(m[2] > max(m[0], m[1]))


if __name__ == "__main__":
    n = raw["pre"].shape[0]
    F = {w: np.zeros((n, 3)) for w in ("pre", "post")}
    with Pool(12) as p:
        for w, k, a, b, c in p.map(feats, [(w, k) for w in ("pre", "post") for k in range(n)]):
            F[w][k] = (a, b, c)
    np.savez(CACHE / "colour_covariates.npz", pre=F["pre"], post=F["post"])
    print("magenta share mean pre/post:", F["pre"][:, 0].mean().round(3), F["post"][:, 0].mean().round(3))
