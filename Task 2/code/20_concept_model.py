"""Concept model: visual concepts (yes/no VLM answers) -> disaster classes via explicit domain rules.

1. Each concept's Yes-vs-No logit is standardised over all pairs (robust median / IQR), so
   'more flood water than a typical pair' is comparable across concepts and the VLM's
   per-question yes-bias cancels. Unlike centring the six class scores, this does not
   assume the six classes are equally frequent.
2. Domain rules (RULES below) turn concepts into class scores. They encode how each disaster
   looks from the air; they were written from disaster knowledge, not fitted to any label.
3. Scores are smoothed over the event graph (pairs of the same event share the label);
   pairs with no visible damage get their label mostly from their event neighbours.
4. Only the Earthquake share is anchored (0.22, the one prior known from the Evaluation
   page); the other class shares are whatever the evidence gives.

usage: python 20_concept_model.py [--tag c1] [--alpha 0.9] [--w_mcq 0.0] [--eq 0.22]
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

# class score = sum(weight * standardised concept). Positive = typical sign of the disaster,
# negative = evidence against it. Key signs get weight 1, supporting context 0.5.
RULES = {
    "Earthquake": {"collapse": 1.0, "city": 0.5, "flood": -0.5, "burn": -0.5, "lava": -0.5, "washed": -0.3,
                   "vegetation": -0.3},
    "Fire": {"burn": 1.0, "vegetation": 0.3, "flood": -0.5, "washed": -0.3, "collapse": -0.3},
    "Hurricane": {"flood": 0.6, "treesdown": 0.6, "roofs": 0.6, "burn": -0.5, "lava": -0.3, "city": -0.3},
    "Tsunami": {"washed": 1.0, "coast": 0.7, "flood": 0.3, "burn": -0.5, "lava": -0.3},
    "Volcano": {"lava": 1.0, "flood": -0.5, "city": -0.3, "coast": -0.2, "roofs": -0.3},
    "Explosion": {"blast": 1.0, "collapse": 0.3, "vegetation": -0.5, "flood": -0.5, "burn": -0.3},
}


def load_concepts(tag="q8b"):
    z = np.load(CACHE / f"concepts_{tag}.npz")
    return z["logit"].astype(np.float64), [str(n) for n in z["names"]]


def standardise(L):
    m = np.isfinite(L[:, 0])
    med = np.median(L[m], 0)
    iqr = np.subtract(*np.percentile(L[m], [75, 25], 0)).clip(1e-3)
    return (L - med) / (iqr / 1.349), med, iqr  # IQR/1.349 ~ std for a normal distribution


def rule_matrix(names):
    W = np.zeros((6, len(names)))
    for k, c in enumerate(CLASSES):
        for n, w in RULES[c].items():
            W[k, names.index(n)] = w
    return W / np.abs(W).sum(1, keepdims=True)  # every class gets the same total weight


def class_scores(Zc, names):
    return np.nan_to_num(Zc) @ rule_matrix(names).T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="c1")
    ap.add_argument("--concepts", default="q8b")
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--eq", type=float, default=0.22)
    ap.add_argument("--w_mcq", type=float, default=0.0, help="weight of the old 8B six-way answer (raw log-probs)")
    a = ap.parse_args()

    L, names = load_concepts(a.concepts)
    scored = np.isfinite(L[:, 0])
    Zc, med, iqr = standardise(L)
    S = class_scores(Zc, names)
    if a.w_mcq:
        lp = np.load(CACHE / "vlm_q8b_clean512.npz")["lp"]
        S = S + a.w_mcq * (lp - lp.mean()) / lp.std()
    # evidence weight: visible damage (low 'nodamage'), and not hidden by clouds
    nd = np.nan_to_num(Zc[:, names.index("nodamage")])
    conf = (1 / (1 + np.exp(nd))) * lm.reliability() * scored + 1e-6
    X = lm.event_features()
    F = lm.propagate(lm.knn_graph(X, a.k), S, conf, a.alpha, iters=200)
    F = F / F.std()
    # anchor only the Earthquake share: shift its bias until exactly eq of pairs are Earthquake
    lo, hi = -10.0, 10.0
    for _ in range(60):
        b = (lo + hi) / 2
        G = F.copy()
        G[:, 0] += b
        share = (G.argmax(1) == 0).mean()
        lo, hi = (b, hi) if share < a.eq else (lo, b)
    bias = np.zeros(6)
    bias[0] = (lo + hi) / 2
    pred = (F + bias).argmax(1)
    share = np.bincount(pred, minlength=6) / len(pred)
    print("scored pairs:", scored.sum())
    print("final share:", dict(zip(CLASSES, np.round(share, 3))))
    ids = list_ids()
    (ROOT / "exp" / "submissions").mkdir(exist_ok=True)
    pd.DataFrame({"Id": ids, "Phenomenon": [CLASSES[i] for i in pred]}).to_csv(
        ROOT / "exp" / "submissions" / f"submission_{a.tag}.csv", index=False)
    np.savez(CACHE / f"final_{a.tag}.npz", F=F, S=F + bias, pred=pred)
    np.savez(MODELS / f"concept_model_{a.tag}.npz", med=med, iqr=iqr, W=rule_matrix(names), bias=bias)
    (MODELS / f"concept_model_{a.tag}.json").write_text(json.dumps(dict(
        concepts=a.concepts, names=names, rules=RULES, alpha=a.alpha, k=a.k, eq=a.eq, w_mcq=a.w_mcq,
        classes=CLASSES), indent=2))


if __name__ == "__main__":
    main()
