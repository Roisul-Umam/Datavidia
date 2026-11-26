"""Fit the label model on all zero-shot signals, predict, and write the submission.

usage: python 15_final.py [--tag v1] [--alpha 0.97] [--k 15] [--eq 0.22] [--prior data|uniform]
writes models/label_model_<tag>.npz, exp/cache/final_<tag>.npz, submission_<tag>.csv
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
sc = importlib.import_module("13_selfcheck")

# how each zero-shot signal is produced (used by t2pipe.step_vlm to rebuild missing caches)
PIPELINE = {
    "tile_vlm": [
        {"model": "Qwen/Qwen3-VL-8B-Instruct", "tag": "q8b_clean512", "src": "clean512", "quant": "4bit"},
        {"model": "Qwen/Qwen3-VL-4B-Instruct", "tag": "q4b_gray", "src": "gray", "quant": "none"},
    ],
    "event_vlm": [],
}
TILE_SIGNALS = {"vlm_q8b_clean512": 1.0, "vlm_q4b_gray": 1.0, "clip_rgbn": 0.25, "clip_gray": 0.25}


def event_signal(tags):
    """Broadcast cluster-level VLM log-probs (calibrated across clusters) to tiles."""
    Zs = []
    for t in tags:
        z = np.load(CACHE / f"evlm_{t}.npz")
        lab, lp = z["labels"], z["lp"]
        sizes = np.bincount(lab, minlength=len(lp))
        mu = (lp * sizes[:, None]).sum(0) / sizes.sum()  # tile-weighted mean per class
        C = lp - mu
        Zs.append(C[lab] / C[lab].std())
    return np.mean(Zs, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--alpha", type=float, default=0.95)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--eq", type=float, default=0.22)
    ap.add_argument("--prior", default="cluster")
    ap.add_argument("--event", default="", help="comma list of evlm tags, e.g. q8b_k40,q8b_k80")
    ap.add_argument("--wevent", type=float, default=1.0)
    ap.add_argument("--T", type=float, default=0.05)
    ap.add_argument("--deconf", type=int, default=1, help="regress corruption covariates out of every signal")
    ap.add_argument("--shrink", type=float, default=0.3, help="shrink data-driven class shares toward uniform")
    ap.add_argument("--conf", type=int, default=1, help="weight each pair's own evidence by cloud-free share")
    a = ap.parse_args()

    names = [n for n in TILE_SIGNALS if (CACHE / (f"{n}.npz" if n.startswith("vlm") else f"clipzs_{n.split('_')[1]}.npz")).exists()]
    sig, dmg = lm.load_signals(names)
    for n in names:  # pairs a VLM has not scored yet carry no evidence from it
        sig[n] = np.where(np.isfinite(sig[n]), sig[n], np.nan)
    stats, coefs = {}, {}
    Cov, _ = lm.noise_covariates()
    Z = 0
    for n in names:
        m = np.isfinite(sig[n][:, 0])
        mu = sig[n][m].mean(0)
        sd = (sig[n][m] - mu).std()
        stats[n] = (mu, sd)
        Zn = (sig[n] - mu) / sd
        if a.deconf:
            Zn, coefs[n] = lm.residualize(Zn, Cov)
        Z = Z + TILE_SIGNALS[n] * np.nan_to_num(Zn)
    if a.event:
        Z = Z + a.wevent * event_signal(a.event.split(","))
    X = lm.event_features()
    P = lm.knn_graph(X, a.k)
    conf = lm.reliability() if a.conf else np.ones(len(Z))
    F = lm.propagate(P, Z, conf, a.alpha, iters=200)
    F = F / F.std()

    # class prior: Earthquake share is known (~0.22 from the all-Earthquake sample score of ~0.06);
    # the other shares come from the smoothed scores themselves ('data') or are uniform.
    raw_share = np.bincount(F.argmax(1), minlength=6) / len(F)
    if a.prior == "cluster":
        # event-level shares: label each KMeans cluster by its mean fused score, average over K
        from sklearn.cluster import KMeans
        sh = []
        for K in (20, 40, 80):
            lab = KMeans(K, n_init=4, random_state=0).fit_predict(X)
            cls = np.array([Z[lab == c].mean(0).argmax() for c in range(K)])[lab]
            sh.append(np.bincount(cls, minlength=6) / len(cls))
        raw_share = np.mean(sh, 0)
    if a.prior == "uniform":
        target = np.full(6, (1 - a.eq) / 5)
    else:
        rest = np.delete(raw_share, 0)
        rest = (1 - a.shrink) * rest / rest.sum() + a.shrink / 5  # shrink toward uniform: rare classes must not vanish
        target = np.insert(rest * (1 - a.eq), 0, 0)
    target[0] = a.eq
    b = lm.fit_bias(F, target, a.T)
    S = F + b
    pred = S.argmax(1)
    share = np.bincount(pred, minlength=6) / len(pred)
    print("signals:", names, "event:", a.event or "-")
    print("raw smoothed share:", dict(zip(CLASSES, np.round(raw_share, 3))))
    print("target share:      ", dict(zip(CLASSES, np.round(target, 3))))
    print("final share:       ", dict(zip(CLASSES, np.round(share, 3))))
    for n in names:  # agreement of the final decision with each (calibrated) input signal
        m = np.isfinite(sig[n][:, 0])
        print(f"  agree with {n:18s}: {np.mean(pred[m] == ((sig[n][m] - stats[n][0])).argmax(1)):.3f}")
    ids = list_ids()
    (ROOT / "exp" / "submissions").mkdir(exist_ok=True)
    pd.DataFrame({"Id": ids, "Phenomenon": [CLASSES[i] for i in pred]}).to_csv(
        ROOT / "exp" / "submissions" / f"submission_{a.tag}.csv", index=False)
    np.savez(CACHE / f"final_{a.tag}.npz", F=F, S=S, pred=pred)
    np.savez(MODELS / f"label_model_{a.tag}.npz", bias=b, target=target,
             **{f"mu_{n}": stats[n][0] for n in names}, **{f"sd_{n}": np.array(stats[n][1]) for n in names},
             **{f"coef_{n}": coefs[n] for n in coefs})
    (MODELS / f"label_model_{a.tag}.json").write_text(json.dumps(dict(
        signals={n: TILE_SIGNALS[n] for n in names}, event=a.event, wevent=a.wevent, k=a.k, alpha=a.alpha,
        T=a.T, eq=a.eq, prior=a.prior, shrink=a.shrink, deconf=a.deconf, conf=a.conf, classes=CLASSES,
        pipeline=PIPELINE), indent=2))


if __name__ == "__main__":
    main()
