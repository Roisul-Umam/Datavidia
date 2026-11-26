"""Label model v2 (after LB 0.305 error analysis).

Changes vs v1, each motivated by a diagnosed failure:
  * six-way VLM answers are calibrated on content-free inputs (contextual calibration)
    instead of subtracting each class's dataset mean, which assumed equal class sizes and
    moved ~600 'Fire' pairs to Volcano;
  * the 4B-on-grayscale signal is dropped: without colour, flood water and burn scars look
    like lava, so its Volcano bias is structural (it answered Volcano for 47% of pairs);
  * grounded yes/no concepts are added; their 'general damage' factor (the VLM says yes to
    every damage question once anything is damaged) is removed so only the TYPE of damage
    counts, and the amount of visible damage becomes the pair's evidence weight;
  * only the Earthquake share is anchored (0.22). A full target can be supplied with
    --target if the real class shares become known.

usage: python 22_v2.py --tag v2a [--w_mcq 1] [--w_con 1] [--alpha 0.9] [--target "0.22,..."]
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import CACHE, CLASSES, MODELS, ROOT, list_ids  # noqa: E402

lm = importlib.import_module("12_label_model")
cm = importlib.import_module("20_concept_model")

DAMAGE = ["flood", "burn", "lava", "collapse", "treesdown", "roofs", "washed", "blast"]


def mcq_scores(tag="vlm_q8b_clean512", cf="q8b", mode="contentfree"):
    lp = np.load(CACHE / f"{tag}.npz")["lp"].astype(np.float64)
    if mode.startswith("contentfree"):
        z = np.load(CACHE / f"contentfree_{cf}.npz")
        rows = [i for i, n in enumerate(z["names"]) if mode == "contentfree" or str(n).startswith("blur")]
        b = z["lp"][rows].mean(0)   # 'contentfree_blur': fully blurred dataset tiles (keep global colour only)
        Z = lp - b
    elif mode == "center":
        Z = lp - lp.mean(0)
    else:
        Z = lp
    return Z / Z.std()


def concept_scores(tag="q8b"):
    L, names = cm.load_concepts(tag)
    Z, _, _ = cm.standardise(L)
    Z = np.nan_to_num(Z)
    di = [names.index(n) for n in DAMAGE]
    g = Z[:, di].mean(1)                       # general 'something is damaged' factor
    R = Z.copy()
    R[:, di] = Z[:, di] - g[:, None]           # damage TYPE profile
    S = cm.class_scores(R, names)
    return S / S.std(), g, Z[:, names.index("nodamage")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2a")
    ap.add_argument("--mcq_mode", default="contentfree_blur")
    ap.add_argument("--w_mcq", type=float, default=1.0)
    ap.add_argument("--w_con", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--eq", type=float, default=0.22)
    ap.add_argument("--target", default="", help="comma list of 6 class shares (overrides --eq)")
    a = ap.parse_args()

    S = 0
    if a.w_mcq:
        S = S + a.w_mcq * mcq_scores(mode=a.mcq_mode)
    con, g, nd = concept_scores()
    if a.w_con:
        S = S + a.w_con * con
    # evidence weight: how visibly damaged the pair is (concepts), and not hidden by clouds
    vis = 1 / (1 + np.exp(-(g - nd) / 2))
    conf = (0.2 + 0.8 * vis) * lm.reliability()
    X = lm.event_features()
    F = lm.propagate(lm.knn_graph(X, a.k), S, conf, a.alpha, iters=200)
    F = F / F.std()
    raw_share = np.bincount(F.argmax(1), minlength=6) / len(F)
    if a.target:
        target = np.array([float(x) for x in a.target.split(",")])
        bias = lm.fit_bias(F, target / target.sum())
    else:  # anchor Earthquake only: bisection on its bias
        lo, hi = -10.0, 10.0
        for _ in range(60):
            b0 = (lo + hi) / 2
            G = F.copy()
            G[:, 0] += b0
            lo, hi = (b0, hi) if (G.argmax(1) == 0).mean() < a.eq else (lo, b0)
        bias = np.zeros(6)
        bias[0] = (lo + hi) / 2
    pred = (F + bias).argmax(1)
    print(f"[{a.tag}] before anchoring:", " ".join(f"{c[:4]}{v:.3f}" for c, v in zip(CLASSES, raw_share)))
    print(f"[{a.tag}] final share     :", " ".join(f"{c[:4]}{v:.3f}" for c, v in zip(CLASSES, np.bincount(pred, minlength=6) / len(pred))))
    ids = list_ids()
    pd.DataFrame({"Id": ids, "Phenomenon": [CLASSES[i] for i in pred]}).to_csv(
        ROOT / "exp" / "submissions" / f"submission_{a.tag}.csv", index=False)
    np.savez(CACHE / f"final_{a.tag}.npz", F=F, S=F + bias, pred=pred)
    np.savez(MODELS / f"label_model_{a.tag}.npz", bias=bias)
    (MODELS / f"label_model_{a.tag}.json").write_text(json.dumps(vars(a), indent=2))


if __name__ == "__main__":
    main()
