"""Pilot helper: make a fixed random subset of pair indices, and render VLM results on it.

usage: python 05b_vlm_pilot.py make [n=96]           -> exp/cache/pilot_idx.npy
       python 05b_vlm_pilot.py show <tag> [src]      -> stats + exp/viz/pilot_<tag>_*.jpg
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import CACHE, CLASSES, VIZ, list_ids, montage  # noqa: E402

if __name__ == "__main__":
    if sys.argv[1] == "make":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 96
        np.save(CACHE / "pilot_idx.npy", np.random.default_rng(42).choice(3180, n, replace=False))
        sys.exit()
    tag = sys.argv[2]
    src = sys.argv[3] if len(sys.argv) > 3 else "canon512"
    spec = importlib.util.spec_from_file_location("vlm", Path(__file__).parent / "05_vlm_zeroshot.py")
    vlm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vlm)
    ids = list_ids()
    idx = np.load(CACHE / "pilot_idx.npy")
    z = np.load(CACHE / f"vlm_{tag}.npz")
    p, dmg = np.exp(z["lp"][idx]), z["dmg"][idx]
    print("argmax share:", dict(zip(CLASSES, np.bincount(p.argmax(1), minlength=6).tolist())))
    print("mean prob:   ", dict(zip(CLASSES, np.round(p.mean(0), 3).tolist())))
    print("dmg quantiles:", np.round(np.quantile(dmg, [.1, .25, .5, .75, .9]), 2).tolist())
    get_pre, get_post = vlm.load_src(src, True)
    items = []
    for k, i in enumerate(idx[:48]):
        top = np.argsort(-p[k])[:2]
        lab = f"{ids[i][:6]} {CLASSES[top[0]][:5]}{p[k, top[0]]:.2f} {CLASSES[top[1]][:4]}{p[k, top[1]]:.2f} d{dmg[k]:.2f}"
        items.append((lab, get_pre(i), get_post(i)))
    for s in range(0, len(items), 16):
        montage(items[s:s + 16], VIZ / f"pilot_{tag}_{s // 16}.jpg", thumb=170)
