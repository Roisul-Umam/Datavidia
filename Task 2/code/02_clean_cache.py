"""Run the cleaning chain once and cache the results as memmaps.

exp/cache/clean512_{pre,post}.u8   (N, 512, 512, 3) uint8  cleaned, colour-normalised RGB
exp/cache/raw512_{pre,post}.u8     (N, 512, 512, 3) uint8  impulse+stripe cleaned, original colour
exp/cache/valid64_{pre,post}.npy   (N, 64, 64) float32     fraction of real (non-padding) pixels per cell
exp/cache/cloud64_{pre,post}.npy   (N, 64, 64) float32     fraction of cloud pixels per cell
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import (CACHE, IMG_DIR, cloud_mask, colour_normalise, destripe, list_ids, read_rgb,
                    remove_impulses, valid_mask)

S = 512


def work(args):
    k, iid, which = args
    rgb = read_rgb(IMG_DIR / iid / f"{which}.jpg")
    valid = valid_mask(rgb)
    x = remove_impulses(rgb)
    x = destripe(x, valid, axis=0)
    x = cv2.resize(x, (S, S), interpolation=cv2.INTER_AREA)
    vm = cv2.resize(valid.astype(np.uint8), (S, S), interpolation=cv2.INTER_NEAREST) > 0
    raw = x.copy()
    xn = colour_normalise(x, vm)
    cm = cloud_mask(xn, vm)
    if (~vm).any() and vm.any():
        xn[~vm] = xn[vm].mean(axis=0).astype(np.uint8)
        raw[~vm] = raw[vm].mean(axis=0).astype(np.uint8)
    v64 = cv2.resize(vm.astype(np.float32), (64, 64), interpolation=cv2.INTER_AREA)
    c64 = cv2.resize(cm.astype(np.float32), (64, 64), interpolation=cv2.INTER_AREA)
    return k, which, xn, raw, v64, c64


if __name__ == "__main__":
    ids = list_ids()
    n = len(ids)
    out = {}
    for w in ("pre", "post"):
        out[w] = dict(
            clean=np.lib.format.open_memmap(CACHE / f"clean512_{w}.npy", "w+", np.uint8, (n, S, S, 3)),
            raw=np.lib.format.open_memmap(CACHE / f"raw512_{w}.npy", "w+", np.uint8, (n, S, S, 3)),
            v=np.zeros((n, 64, 64), np.float32), c=np.zeros((n, 64, 64), np.float32))
    jobs = [(k, i, w) for k, i in enumerate(ids) for w in ("pre", "post")]
    t = time.time()
    with Pool(12) as pool:
        for j, (k, w, xn, raw, v64, c64) in enumerate(pool.imap_unordered(work, jobs, chunksize=8)):
            out[w]["clean"][k] = xn
            out[w]["raw"][k] = raw
            out[w]["v"][k] = v64
            out[w]["c"][k] = c64
            if j % 1000 == 0:
                print(j, f"{time.time()-t:.0f}s", flush=True)
    for w in ("pre", "post"):
        out[w]["clean"].flush()
        out[w]["raw"].flush()
        np.save(CACHE / f"valid64_{w}.npy", out[w]["v"])
        np.save(CACHE / f"cloud64_{w}.npy", out[w]["c"])
    print(f"done in {time.time()-t:.0f}s")
