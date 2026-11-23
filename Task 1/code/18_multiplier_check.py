"""Uji prosedur pengali kelas secara out-of-fold sebelum dipakai untuk submission.

Semua prosedur dipelajari tanpa label fold yang dinilai.
A: label, grid kasar, 400 ribu baris (prosedur yang dinilai di 17_blend.py).
B: label, grid kasar lalu halus, 72% baris (prosedur final di 17_blend.py).
C: label, rata-rata log pengali dari 7 sampel 250 ribu baris.
D: tanpa label, memaksimalkan F1 harapan dari probabilitas baris yang diprediksi.
E: label, pengali terpisah untuk baris dengan dan tanpa riwayat.

Prosedur yang lebih rumit dari A hanya dipakai jika F1 naik lebih dari 0,001 dan menang di minimal 4 dari 5 fold.
"""
import json
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, r"E:\Kuliyah\Datavidia\Task 1\code")
from common import apply_weights, f1_from_pred, optimize_class_weights
from features import BASE, CLASSES, K

RUN = BASE / "exp" / os.environ.get("RUN_NAME", "v2")
N_FOLDS = 5
MIN_GAIN = 0.001
MIN_FOLDS = 4
COARSE = np.round(np.arange(0.5, 4.01, 0.05), 3)
FINE = np.round(np.arange(0.5, 4.01, 0.02), 3)
t0 = time.time()


def log(msg):
    print(f"[{(time.time() - t0) / 60:6.1f}m] {msg}", flush=True)


y = np.load(RUN / "y.npy").astype(np.int64)
fold = np.load(RUN / "fold_id.npy")
Pm = np.load(RUN / "oof_blend_testlike.npy")
Ps = np.load(RUN / "test_blend.npy")
n = len(y)
gm = np.zeros(n, dtype=np.int8)
for f in range(N_FOLDS):
    gm[np.load(RUN / f"val_idx_f{f}.npy")] = np.load(RUN / f"val_blank_f{f}.npy").astype(np.int8)
gs = (pl.read_parquet(BASE / "exp" / "cache" / "fd_test.parquet", columns=["drv_n"])["drv_n"].to_numpy() == 0).astype(np.int8)
log(f"tanpa riwayat: validasi {gm.mean():.4f} | test {gs.mean():.4f}")


def expected_f1(P, pred, sp=None):
    sp = P.sum(0).astype(np.float64) if sp is None else sp
    tp = np.bincount(pred, weights=P[np.arange(len(pred)), pred], minlength=K)
    return float((2 * tp / (np.bincount(pred, minlength=K) + sp)).mean())


def fit_expected(P, passes=4):
    lp = np.log(np.clip(P, 1e-9, 1)).astype(np.float32)
    sp = P.sum(0).astype(np.float64)
    w = np.zeros(K, np.float32)
    best = expected_f1(P, lp.argmax(1), sp)
    lg = np.log(COARSE).astype(np.float32)
    for _ in range(passes):
        moved = False
        for c in range(1, K):
            o = lp + w
            o[:, c] = -np.inf
            om, oa = o.max(1), o.argmax(1)
            bs, bv = best, w[c]
            for g in lg:
                s = expected_f1(P, np.where(lp[:, c] + g > om, c, oa), sp)
                if s > bs + 1e-7:
                    bs, bv = s, g
            if bv != w[c]:
                w[c], best, moved = bv, bs, True
        if not moved:
            break
    return np.exp(w).astype(np.float64)


def fit_groups(P, t, g, passes=3):
    lp = np.log(np.clip(P, 1e-9, 1)).astype(np.float32)
    W = np.zeros((2, K), np.float32)
    lg = np.log(COARSE).astype(np.float32)
    pred = lp.argmax(1)
    best = f1_from_pred(t, pred)[0]
    for _ in range(passes):
        moved = False
        for h in (0, 1):
            m = g == h
            for c in range(1, K):
                o = lp[m] + W[h]
                o[:, c] = -np.inf
                om, oa = o.max(1), o.argmax(1)
                bs, bv, bp = best, W[h, c], None
                for v in lg:
                    p = pred.copy()
                    p[m] = np.where(lp[m, c] + v > om, c, oa)
                    s = f1_from_pred(t, p)[0]
                    if s > bs + 1e-6:
                        bs, bv, bp = s, v, p
                if bp is not None:
                    W[h, c], best, pred, moved = bv, bs, bp, True
        if not moved:
            break
    return np.exp(W).astype(np.float64)


def apply_groups(P, W, g):
    return (np.log(np.clip(P, 1e-9, 1)) + np.log(W[g])).argmax(1)


def fit_a(idx, rs):
    s = rs.choice(idx, 400_000, replace=False)
    return optimize_class_weights(Pm[s], y[s], passes=3)[0]


def fit_b(idx, rs):
    s = rs.choice(idx, int(0.72 * len(idx)), replace=False)
    w, _ = optimize_class_weights(Pm[s], y[s], passes=3)
    return optimize_class_weights(Pm[s], y[s], passes=2, w0=w, grid=FINE)[0]


def fit_c(idx, rs):
    lw = [np.log(optimize_class_weights(Pm[s], y[s], passes=3)[0]) for s in (rs.choice(idx, 250_000, replace=False) for _ in range(7))]
    return np.exp(np.mean(lw, axis=0))


def fit_e(idx, rs):
    s = rs.choice(idx, 400_000, replace=False)
    return fit_groups(Pm[s], y[s], gm[s])


names = ["A", "B", "C", "D", "E"]
oof = {k: np.zeros(n, dtype=np.int8) for k in names}
per_fold = {k: [] for k in names}
for f in range(N_FOLDS):
    tr, va = np.where(fold != f)[0], np.where(fold == f)[0]
    rs = np.random.RandomState(700 + f)
    oof["A"][va] = apply_weights(Pm[va], fit_a(tr, rs))
    oof["B"][va] = apply_weights(Pm[va], fit_b(tr, rs))
    oof["C"][va] = apply_weights(Pm[va], fit_c(tr, rs))
    oof["D"][va] = apply_weights(Pm[va], fit_expected(Pm[va]))
    oof["E"][va] = apply_groups(Pm[va], fit_e(tr, rs), gm[va])
    for k in names:
        per_fold[k].append(round(f1_from_pred(y[va], oof[k][va])[0], 5))
    log(f"fold {f}: " + " | ".join(f"{k}={per_fold[k][-1]:.5f}" for k in names))

rs = np.random.RandomState(777)
idx_all = np.arange(n)
final = {
    "A": fit_a(idx_all, rs),
    "B": fit_b(idx_all, rs),
    "C": fit_c(idx_all, rs),
    "D": fit_expected(Ps),
    "E": fit_e(idx_all, rs),
}
test_pred = {k: (apply_groups(Ps, final[k], gs) if k == "E" else apply_weights(Ps, final[k])).astype(np.int8) for k in names}

ids = pl.read_parquet(BASE / "exp" / "cache" / "fe_test.parquet", columns=["shipment_id"])
cur = ids.join(pl.read_csv(BASE / "exp" / "submissions" / "submission_v2.csv"), on="shipment_id", how="left", maintain_order="left")
cur = cur["status_penanganan"].replace_strict(CLASSES, list(range(K)), return_dtype=pl.Int8).to_numpy().astype(np.int64)
base = np.array(per_fold["A"])
result = {}
for k in names:
    score, per_class = f1_from_pred(y, oof[k])
    delta = np.array(per_fold[k]) - base
    result[k] = dict(
        f1_oof=round(score, 5),
        f1_per_fold=per_fold[k],
        selisih_vs_A=round(score - f1_from_pred(y, oof["A"])[0], 5),
        fold_menang_vs_A=int((delta > 0).sum()),
        f1_harapan_oof=round(expected_f1(Pm, oof[k].astype(np.int64)), 5),
        f1_tanpa_riwayat=round(f1_from_pred(y[gm == 1], oof[k][gm == 1])[0], 5),
        f1_dengan_riwayat=round(f1_from_pred(y[gm == 0], oof[k][gm == 0])[0], 5),
        f1_per_kelas=dict(zip(CLASSES, np.round(per_class, 4).tolist())),
        pengali_final=np.round(final[k], 3).tolist(),
        f1_harapan_test=round(expected_f1(Ps, test_pred[k].astype(np.int64)), 5),
        sebaran_test=dict(zip(CLASSES, np.round(np.bincount(test_pred[k], minlength=K) / len(Ps), 4).tolist())),
        cocok_dgn_submission_v2=round(float((test_pred[k] == cur).mean()), 4),
    )

chosen = "A"
candidates = [k for k in names[1:] if result[k]["selisih_vs_A"] > MIN_GAIN and result[k]["fold_menang_vs_A"] >= MIN_FOLDS]
if candidates:
    chosen = max(candidates, key=lambda k: result[k]["f1_oof"])
report = dict(terpilih=chosen, sebaran_train=dict(zip(CLASSES, np.round(np.bincount(y, minlength=K) / n, 4).tolist())), prosedur=result, menit=round((time.time() - t0) / 60, 1))
(RUN / "multiplier_check.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
np.savez_compressed(RUN / "multiplier_test_preds.npz", **test_pred)
print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
