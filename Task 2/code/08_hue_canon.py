"""Undo the organisers' hue jitter (purple/magenta/blue vegetation) without labels.

Model: each colour image I was produced from a natural image by a hue rotation by theta,
possibly composed with a red/blue swap (a hue reflection). For every image we compute
a saturation-weighted hue histogram h (72 bins of 5 deg). A 'natural' hue prior P is
learned from the data itself by EM-style alternation:
  1. theta_i, r_i = argmax over 72 rotations x 2 reflections of  sum_b h_i(b) log P(T(b))
  2. P = mean of the corrected histograms (smoothed)
Natural aerial scenes concentrate hue in yellow-green (vegetation) and orange (soil/roofs),
so this converges to that prior. Pre and post of a pair are fitted jointly: their
corrected histograms should agree, which adds a consistency term.

writes exp/cache/hue_params.npz (theta_pre, refl_pre, theta_post, refl_post, conf_*)
       exp/cache/canon512_{pre,post}.npy (hue-corrected cleaned images)
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import CACHE, VIZ, list_ids, montage  # noqa: E402

NB = 72  # bins of 5 degrees (OpenCV hue units 0..180 -> 2.5 units per bin)


def hue_hist(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h = hsv[..., 0].astype(np.float32) * 2  # degrees
    s = hsv[..., 1].astype(np.float32) / 255
    v = hsv[..., 2].astype(np.float32) / 255
    w = s * (v > 0.08) * (v < 0.97)
    hist = np.bincount((h // (360 / NB)).astype(int).ravel() % NB, weights=w.ravel(), minlength=NB)
    return hist, w.mean()


def candidates(hist):
    """All 144 transformed versions of a histogram: rotate by k bins, optionally reflect.

    Reflection = red/blue swap: hue h -> 240 - h (degrees).
    """
    out = []
    refl = np.array([hist[int(((240 - (b + 0.5) * 5) % 360) // 5)] for b in range(NB)])
    for r, base in enumerate((hist, refl)):
        for k in range(NB):
            out.append(np.roll(base, -k))  # correction: shift hue by -k bins
    return np.array(out)  # (144, NB)


def apply_correction(rgb, k, refl):
    """Inverse of the jitter: first optional reflection, then rotate hue by -k*5 deg."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.int32)
    h = hsv[..., 0] * 2
    if refl:
        h = (240 - h) % 360
    h = (h - k * 5) % 360
    hsv[..., 0] = (h // 2).astype(np.int32)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def hist_job(args):
    k, which = args
    arr = np.load(CACHE / f"raw512_{which}.npy", mmap_mode="r")
    h, sw = hue_hist(np.asarray(arr[k]))
    return k, which, h, sw


def fit(H_pre, H_post, iters=6):
    n = len(H_pre)
    norm = lambda H: H / H.sum(-1, keepdims=True).clip(1e-9)
    Hp, Hq = norm(H_pre), norm(H_post)
    Cp = np.stack([candidates(h) for h in Hp])  # (n,144,NB)
    Cq = np.stack([candidates(h) for h in Hq])
    P = (Hp.sum(0) + Hq.sum(0))
    for it in range(iters):
        P = norm(P + 1e-3 * P.sum())
        # smooth prior circularly (Gaussian, sigma ~ 2 bins)
        ker = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
        P = np.convolve(np.concatenate([P[-6:], P, P[:6]]), ker / ker.sum(), mode="valid")
        logP = np.log(P / P.sum())
        Lp = Cp @ logP  # (n,144)
        Lq = Cq @ logP
        # joint fit: pre/post corrected histograms should be similar (Bhattacharyya)
        sim = np.einsum("iab,icb->iac", np.sqrt(Cp), np.sqrt(Cq))  # (n,144,144)
        score = Lp[:, :, None] + Lq[:, None, :] + 2.0 * np.log(sim.clip(1e-6))
        best = score.reshape(n, -1).argmax(1)
        bp, bq = best // 144, best % 144
        P = Cp[np.arange(n), bp].sum(0) + Cq[np.arange(n), bq].sum(0)
        print(f"iter {it}: share refl pre {np.mean(bp >= NB):.3f}, share k=0 pre {np.mean(bp == 0):.3f}", flush=True)
    # confidence: margin between best and best alternative that is >= 30 deg away
    def conf(L, b):
        Lc = L.copy()
        k = b % NB
        for d in range(-6, 7):
            Lc[np.arange(n), (b // NB) * NB + (k + d) % NB] = -np.inf
        return L[np.arange(n), b] - Lc.max(1)
    return bp, bq, conf(Lp, bp), conf(Lq, bq), P / P.sum()


def correct_job(args):
    k, which, code = args
    arr = np.load(CACHE / f"raw512_{which}.npy", mmap_mode="r")
    rgb = np.asarray(arr[k])
    return k, which, apply_correction(rgb, code % NB, code >= NB)


if __name__ == "__main__":
    ids = list_ids()
    n = len(ids)
    t = time.time()
    with Pool(12) as pool:
        res = pool.map(hist_job, [(k, w) for k in range(n) for w in ("pre", "post")], chunksize=32)
    H = {"pre": np.zeros((n, NB)), "post": np.zeros((n, NB))}
    S = {"pre": np.zeros(n), "post": np.zeros(n)}
    for k, w, h, sw in res:
        H[w][k], S[w][k] = h, sw
    print(f"hists {time.time()-t:.0f}s")
    bp, bq, cp, cq, P = fit(H["pre"], H["post"])
    gray = (S["pre"] < 0.02) & (S["post"] < 0.02)
    bp[S["pre"] < 0.02] = 0
    bq[S["post"] < 0.02] = 0
    np.savez(CACHE / "hue_params.npz", code_pre=bp, code_post=bq, conf_pre=cp, conf_post=cq, prior=P,
             sat_pre=S["pre"], sat_post=S["post"])
    print("prior peak deg:", (np.argmax(P) + 0.5) * 5, " gray pairs:", gray.sum())
    print("rotation (deg) distribution pre:", np.histogram((bp % NB) * 5, bins=12, range=(0, 360))[0].tolist())
    print("reflection share pre/post:", (bp >= NB).mean().round(3), (bq >= NB).mean().round(3))
    out = {w: np.lib.format.open_memmap(CACHE / f"canon512_{w}.npy", "w+", np.uint8, (n, 512, 512, 3))
           for w in ("pre", "post")}
    with Pool(12) as pool:
        for k, w, img in pool.imap_unordered(correct_job, [(k, w, (bp if w == "pre" else bq)[k])
                                                           for k in range(n) for w in ("pre", "post")], chunksize=16):
            out[w][k] = img
    for w in out:
        out[w].flush()
    print(f"done {time.time()-t:.0f}s")
    # visual check on the most-rotated images
    raw_pre = np.load(CACHE / "raw512_pre.npy", mmap_mode="r")
    can_pre = out["pre"]
    rot = np.minimum((bp % NB), NB - (bp % NB)) + 100 * (bp >= NB)
    pick = np.random.default_rng(0).choice(np.where(rot > 6)[0], 16, replace=False)
    montage([(f"{ids[i][:6]} code{bp[i]} c{cp[i]:.2f}", raw_pre[i], can_pre[i]) for i in pick], VIZ / "hue_canon_check.jpg", thumb=170)
