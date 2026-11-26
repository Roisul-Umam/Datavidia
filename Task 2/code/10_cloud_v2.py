"""Stricter cloud detector: large, bright, unsaturated, smooth blobs with SOFT edges.

v1 (bright+unsaturated+smooth) also fired on white roofs, sand and concrete. Synthetic
clouds differ from those in two ways: they are big, and their borders fade gradually
(alpha blending) while roofs have sharp borders. We keep connected components that are
(a) at least ~0.6% of the image after a 15px opening and (b) have a low mean gradient
along their boundary.
Overwrites exp/cache/cloud64_{pre,post}.npy (old ones kept as cloud64v1_*).
"""
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import CACHE, VIZ, list_ids, montage  # noqa: E402

MIN_AREA = 1500
EDGE_MAX = 9.0


def detect(rgb, valid):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    v, s = hsv[..., 2] / 255, hsv[..., 1] / 255
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mu = cv2.blur(g, (15, 15))
    sd = np.sqrt(np.maximum(cv2.blur(g * g, (15, 15)) - mu * mu, 0))
    m = ((v > 0.70) & (s < 0.22) & (sd < 10) & valid).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    gx = cv2.Sobel(cv2.GaussianBlur(g, (3, 3), 0), cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(cv2.GaussianBlur(g, (3, 3), 0), cv2.CV_32F, 0, 1)
    grad = np.sqrt(gx * gx + gy * gy) / 8
    out = np.zeros_like(m, bool)
    for c in range(1, n):
        if stats[c, cv2.CC_STAT_AREA] < MIN_AREA:
            continue
        comp = (lab == c).astype(np.uint8)
        ring = cv2.dilate(comp, np.ones((5, 5), np.uint8)) - cv2.erode(comp, np.ones((5, 5), np.uint8))
        ring = (ring > 0) & valid
        if ring.any() and grad[ring].mean() > EDGE_MAX:
            continue  # sharp border -> roof / building / road, not a cloud
        out |= comp > 0
    # clouds fade out: grow the core a little to cover the semi-transparent rim
    out = cv2.dilate(out.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0
    return out & valid


def job(args):
    k, w = args
    rgb = np.asarray(np.load(CACHE / f"clean512_{w}.npy", mmap_mode="r")[k])
    va = cv2.resize(np.load(CACHE / f"valid64_{w}.npy", mmap_mode="r")[k], (512, 512), interpolation=cv2.INTER_LINEAR) > 0.5
    m = detect(rgb, va)
    return k, w, cv2.resize(m.astype(np.float32), (64, 64), interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    n = len(list_ids())
    for w in ("pre", "post"):
        if not (CACHE / f"cloud64v1_{w}.npy").exists():
            shutil.copy(CACHE / f"cloud64_{w}.npy", CACHE / f"cloud64v1_{w}.npy")
    out = {w: np.zeros((n, 64, 64), np.float32) for w in ("pre", "post")}
    with Pool(12) as pool:
        for k, w, c in pool.imap_unordered(job, [(k, w) for k in range(n) for w in ("pre", "post")], chunksize=16):
            out[w][k] = c
    for w in out:
        np.save(CACHE / f"cloud64_{w}.npy", out[w])
        old = np.load(CACHE / f"cloud64v1_{w}.npy")
        print(w, "images with >5% cloud: v1", (old.mean((1, 2)) > .05).mean().round(3), " v2", (out[w].mean((1, 2)) > .05).mean().round(3),
              "| any cloud: v1", (old.max((1, 2)) > .25).mean().round(3), " v2", (out[w].max((1, 2)) > .25).mean().round(3))
    # visual check: images where v1 fired but v2 did not, and images v2 keeps
    ids = list_ids()
    arr = np.load(CACHE / "clean512_post.npy", mmap_mode="r")
    old = np.load(CACHE / "cloud64v1_post.npy")
    dropped = np.where((old.mean((1, 2)) > .03) & (out["post"].mean((1, 2)) < .005))[0]
    kept = np.where(out["post"].mean((1, 2)) > .05)[0]
    rng = np.random.default_rng(0)

    def overlay(i, m64):
        x = np.asarray(arr[i]).copy()
        m = cv2.resize(m64, (512, 512)) > 0.25
        x[m] = (0.5 * x[m] + [127, 0, 0]).astype(np.uint8)
        return x
    items = [(f"drop {ids[i][:6]}", overlay(i, old[i]), overlay(i, out['post'][i])) for i in rng.choice(dropped, 8, replace=False)]
    items += [(f"keep {ids[i][:6]}", overlay(i, old[i]), overlay(i, out['post'][i])) for i in rng.choice(kept, 8, replace=False)]
    montage(items, VIZ / "cloud_v2_check.jpg", thumb=170)
