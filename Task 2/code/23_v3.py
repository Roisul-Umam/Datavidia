"""Label model v3: calibrated six-way answers + two batches of grounded concepts + event graph.

Concept rules (RULES) are domain knowledge, written before seeing any score of this model:
  * Fire needs burn evidence; plain vegetation is NOT evidence for Fire (v2 did that and
    turned an unburned tropical forest island into Fire); wildfires with satellite coverage
    are in dry climates, so a tropical scene counts slightly against Fire;
  * a disaster's visible signature (burn scars -> Fire, lava/ash -> Volcano, collapsed
    buildings in a city -> Earthquake, shore washed away -> Tsunami, flood water / fallen
    trees / damaged roofs -> Hurricane, point-centred blast zone -> Explosion);
  * general geography: a tsunami needs open sea in the BEFORE image and practically never
    hits the US mainland, while large US floods and roof/tree damage are hurricane-driven;
    the earthquakes and explosions with satellite coverage are outside the US.
The 'general damage' factor of the damage concepts is removed (the VLM tends to say yes to
every damage question once anything is damaged) so the TYPE of damage decides, and the
amount of visible damage becomes the pair's evidence weight.

usage: python 23_v3.py --tag v3 [--w_mcq 1] [--w_con 1] [--alpha 0.9] [--eq 0.22] [--target ...]
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
v2 = importlib.import_module("22_v2")

DAMAGE = ["flood", "burn", "lava", "collapse", "treesdown", "roofs", "washed", "blast",
          "inlandflood", "lavaflow", "burnscar"]
RULES = {
    "Earthquake": {"collapse": 1.0, "city": 0.5, "snow": 0.3, "usa": -0.5, "flood": -0.4, "inlandflood": -0.3,
                   "burn": -0.4, "lavaflow": -0.4, "washed": -0.3, "vegetation": -0.3},
    "Fire": {"burn": 0.7, "burnscar": 0.7, "flood": -0.4, "inlandflood": -0.4, "tropical": -0.3,
             "washed": -0.3, "collapse": -0.3, "snow": -0.3},
    "Hurricane": {"flood": 0.4, "inlandflood": 0.4, "treesdown": 0.6, "roofs": 0.6, "usa": 0.5,
                  "burn": -0.4, "burnscar": -0.3, "lavaflow": -0.3, "city": -0.3, "snow": -0.3},
    "Tsunami": {"washed": 1.0, "coastpre": 1.0, "flood": 0.3, "usa": -0.7, "burn": -0.4, "burnscar": -0.3,
                "lavaflow": -0.3, "snow": -0.3},
    "Volcano": {"lavaflow": 1.0, "lava": 0.5, "tropical": 0.2, "flood": -0.4, "inlandflood": -0.4,
                "city": -0.3, "roofs": -0.3, "snow": -0.2},
    "Explosion": {"blast": 1.0, "collapse": 0.3, "city": 0.3, "usa": -0.5, "vegetation": -0.5, "flood": -0.4,
                  "burn": -0.3, "snow": -0.3},
}


def load_all_concepts():
    L1, n1 = cm.load_concepts("q8b")
    L2, n2 = cm.load_concepts("q8b_set2")
    return np.hstack([L1, L2]), n1 + n2


def concept_scores():
    L, names = load_all_concepts()
    Z, med, iqr = cm.standardise(L)
    Z = np.nan_to_num(Z)
    di = [names.index(n) for n in DAMAGE]
    g = Z[:, di].mean(1)
    R = Z.copy()
    R[:, di] = Z[:, di] - g[:, None]
    W = np.zeros((6, len(names)))
    for k, c in enumerate(CLASSES):
        for n, w in RULES[c].items():
            W[k, names.index(n)] = w
    W /= np.abs(W).sum(1, keepdims=True)
    S = R @ W.T
    return S / S.std(), g, Z[:, names.index("nodamage")], dict(med=med, iqr=iqr, W=W, names=names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v3")
    ap.add_argument("--w_mcq", type=float, default=1.0)
    ap.add_argument("--w_con", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--eq", type=float, default=0.22)
    ap.add_argument("--target", default="")
    a = ap.parse_args()

    S = 0
    if a.w_mcq:
        S = S + a.w_mcq * v2.mcq_scores(mode="contentfree_blur")
    con, g, nd, par = concept_scores()
    if a.w_con:
        S = S + a.w_con * con
    vis = 1 / (1 + np.exp(-(g - nd) / 2))
    conf = (0.2 + 0.8 * vis) * lm.reliability()
    X = lm.event_features()
    F = lm.propagate(lm.knn_graph(X, a.k), S, conf, a.alpha, iters=200)
    F = F / F.std()
    raw_share = np.bincount(F.argmax(1), minlength=6) / len(F)
    if a.target:
        t = np.array([float(x) for x in a.target.split(",")])
        bias = lm.fit_bias(F, t / t.sum())
    else:
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
    np.savez(MODELS / f"label_model_{a.tag}.npz", bias=bias, concept_med=par["med"], concept_iqr=par["iqr"],
             W=par["W"], cf_bias=np.load(CACHE / "contentfree_q8b.npz")["lp"])
    (MODELS / f"label_model_{a.tag}.json").write_text(json.dumps(dict(vars(a), rules=RULES, damage=DAMAGE,
                                                                      concept_names=par["names"]), indent=2))


if __name__ == "__main__":
    main()
