"""Label-free model selection: predict a HELD-OUT zero-shot signal from the others.

If signals are noisy views of the same true label with roughly independent errors, a label
model that is closer to the truth agrees more with an independent held-out signal. We use
per-class Spearman correlation between the model's calibrated scores and the held-out
signal's calibrated scores (soft), plus argmax agreement (hard), on pairs scored by all.

usage: python 13_selfcheck.py <heldout> <sig1> [<sig2> ...]
"""
import importlib
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
lm = importlib.import_module("12_label_model")


def score(F, H, m):
    Hc = lm.calibrate(H, m)
    rc = np.mean([spearmanr(F[m, k], Hc[m, k]).correlation for k in range(6)])
    agree = np.mean(F[m].argmax(1) == Hc[m].argmax(1))
    return rc, agree


if __name__ == "__main__":
    held, used = sys.argv[1], sys.argv[2:]
    sig, dmg = lm.load_signals([held] + used)
    m = np.all([np.isfinite(sig[n][:, 0]) for n in sig], axis=0)
    print("pairs scored by all signals:", m.sum())
    X = lm.event_features()
    conf_opts = {"none": None}
    if dmg is not None:
        conf_opts["dmg"] = np.nan_to_num(dmg, nan=0.5)
        conf_opts["dmg^2"] = conf_opts["dmg"] ** 2 + 0.05
    sub = {n: np.where(m[:, None], sig[n], np.nan) for n in used}
    for n in used:  # restrict calibration statistics to the common subset
        sub[n] = np.where(np.isfinite(sub[n]), sub[n], 0.0)
    rc, ag = score(sum(lm.calibrate(sig[n], m) for n in used), sig[held], m)
    print(f"raw (no graph)                 rank={rc:.3f} agree={ag:.3f}")
    for k in (10, 20):
        for alpha in (0.5, 0.8, 0.9, 0.95):
            for cn, conf in conf_opts.items():
                Z = sum(lm.calibrate(sig[n], m) * m[:, None] for n in used)
                c = (np.ones(len(Z)) if conf is None else conf) * m  # unscored pairs carry no evidence
                F = lm.propagate(lm.knn_graph(X, k), Z, c + 1e-6, alpha)
                rc, ag = score(F, sig[held], m)
                print(f"k={k:2d} alpha={alpha:.2f} conf={cn:6s} rank={rc:.3f} agree={ag:.3f}", flush=True)
