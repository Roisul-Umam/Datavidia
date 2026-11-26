"""Tune label-model settings by agreement with a held-out, independently produced signal.

Held-out: the 4B VLM on grayscale images (different model AND different view from the
8B colour run). It is deconfounded the same way, so shared artefact biases (clouds ->
smoke) do not inflate agreement.
"""
import importlib
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
lm = importlib.import_module("12_label_model")
sc = importlib.import_module("13_selfcheck")

HELD = "vlm_q4b_gray"
USED = {"vlm_q8b_clean512": 1.0, "clip_rgbn": 0.35, "clip_gray": 0.35}


def calibrated(sig, n, C, deconf):
    S = sig[n]
    m = np.isfinite(S[:, 0])
    Z = (S - S[m].mean(0)) / (S[m] - S[m].mean(0)).std()
    if deconf:
        Z, _ = lm.residualize(Z, C)
    return Z


if __name__ == "__main__":
    sig, _ = lm.load_signals([HELD] + list(USED))
    m = np.isfinite(sig[HELD][:, 0])
    print("held-out pairs:", m.sum())
    C, _ = lm.noise_covariates()
    X = lm.event_features()
    P = lm.knn_graph(X, 15)
    rel = lm.reliability()
    for deconf, conf in itertools.product((0, 1), (0, 1)):
        Z = sum(w * calibrated(sig, n, C, deconf) for n, w in USED.items())
        H = calibrated(sig, HELD, C, deconf)
        H = np.where(m[:, None], H, np.nan)
        c = rel if conf else np.ones(len(Z))
        res = [f"raw {sc.score(Z, H, m)[0]:.3f}/{sc.score(Z, H, m)[1]:.3f}"]
        for a in (0.9, 0.95, 0.97, 0.99):
            F = lm.propagate(P, Z, c, a, iters=200)
            rc, ag = sc.score(F, H, m)
            res.append(f"a{a}: {rc:.3f}/{ag:.3f}")
        print(f"deconf={deconf} conf={conf} | " + " | ".join(res), flush=True)
