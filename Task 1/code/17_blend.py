"""Blend hasil 5-fold, pengali kelas cross-fitted, pemeriksaan terhadap skor LB, dan pembuatan submission.

Skor utama memakai prediksi validasi yang meniru test (11,4% baris dikosongkan riwayatnya).
Pengali kelas dipelajari dari 4 fold lalu dinilai pada fold kelima.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize

sys.path.insert(0, r"E:\Kuliyah\Datavidia\Task 1\code")
from common import apply_weights, f1_from_pred, optimize_class_weights
from features import BASE, CLASSES, DATA, K

RUN = BASE / "exp" / os.environ.get("RUN_NAME", "v2")
PRED = RUN / "preds"
OUT_NAME = os.environ.get("OUT_NAME", "submission_v2")
SUB = BASE / "exp" / "submissions"
N_FOLDS = 5
LB_KNOWN = {"submission.csv": 0.59364, "exp/submissions/sub_interim_lgbm.csv": 0.48574}
t0 = time.time()
y = np.load(RUN / "y.npy").astype(np.int64)
fold_id = np.load(RUN / "fold_id.npy")
n = len(y)
models = sorted({p.name.split("_f")[0] for p in PRED.glob("*_f0.npz") if all((PRED / f"{p.name.split('_f')[0]}_f{f}.npz").exists() for f in range(N_FOLDS))})
print("model lengkap 5 fold:", models, flush=True)


def softmax(L):
    L = L - L.max(1, keepdims=True)
    P = np.exp(L)
    return P / P.sum(1, keepdims=True)


def ll(P, t):
    return float(-np.log(np.clip(P[np.arange(len(t)), t], 1e-9, 1)).mean())


Lv, Lm, Ls = {}, {}, {}
for name in models:
    v, m, s = np.zeros((n, K), np.float32), np.zeros((n, K), np.float32), None
    for f in range(N_FOLDS):
        d = np.load(PRED / f"{name}_f{f}.npz")
        idx = np.load(RUN / f"val_idx_f{f}.npy")
        v[idx], m[idx] = d["v"], d["m"]
        ls = np.log(np.clip(d["s"], 1e-9, 1))
        s = ls if s is None else s + ls
    Lv[name], Lm[name], Ls[name] = np.log(np.clip(v, 1e-9, 1)), np.log(np.clip(m, 1e-9, 1)), s / N_FOLDS
    print(f"{name:8s} logloss validasi={ll(v, y):.5f} | meniru test={ll(m, y):.5f} | F1 argmax meniru test={f1_from_pred(y, m.argmax(1))[0]:.5f}", flush=True)

rs = np.random.RandomState(42)
smp = rs.choice(n, 400_000, replace=False)
Lsm = np.stack([Lm[k][smp] for k in models])
def obj(w):
    return ll(softmax(np.tensordot(w, Lsm, axes=(0, 0))), y[smp])
w0 = np.ones(len(models)) / len(models)
res = minimize(obj, w0, method="SLSQP", bounds=[(0, 1)] * len(models), constraints={"type": "eq", "fun": lambda w: w.sum() - 1}, options=dict(maxiter=60, ftol=1e-6))
weights = dict(zip(models, np.round(res.x, 4).tolist()))
print("bobot blend:", weights, "| logloss sampel:", round(res.fun, 5), flush=True)
del Lsm

def blend(store):
    return softmax(sum(weights[k] * store[k] for k in models)).astype(np.float32)

Pv, Pm, Ps = blend(Lv), blend(Lm), blend(Ls)
del Lv, Lm, Ls
print(f"blend: logloss validasi={ll(Pv, y):.5f} | meniru test={ll(Pm, y):.5f} | F1 argmax meniru test={f1_from_pred(y, Pm.argmax(1))[0]:.5f}", flush=True)

pred_cf, per_fold = np.zeros(n, dtype=np.int8), []
for f in range(N_FOLDS):
    o, v = np.where(fold_id != f)[0], np.where(fold_id == f)[0]
    fit = rs.choice(o, 400_000, replace=False)
    w, _ = optimize_class_weights(Pm[fit], y[fit], passes=3)
    pred_cf[v] = apply_weights(Pm[v], w)
    per_fold.append(round(f1_from_pred(y[v], pred_cf[v])[0], 5))
    print(f"  fold {f}: F1 meniru test = {per_fold[-1]} | pengali {np.round(w, 2).tolist()}", flush=True)
f1_cf, per_class = f1_from_pred(y, pred_cf)
print(f"Macro F1 cross-fitted meniru test = {f1_cf:.5f}", flush=True)

fit = rs.choice(n, 900_000, replace=False)
w_final, _ = optimize_class_weights(Pm[fit], y[fit], passes=3)
w_final, _ = optimize_class_weights(Pm[fit], y[fit], passes=2, w0=w_final, grid=np.round(np.arange(0.5, 4.01, 0.02), 3))
pred_s = apply_weights(Ps, w_final)

ids = pl.read_parquet(BASE / "exp" / "cache" / "fe_test.parquet", columns=["shipment_id"])
def to_idx(path):
    d = ids.join(pl.read_csv(path), on="shipment_id", how="left", maintain_order="left")
    return d["status_penanganan"].replace_strict(CLASSES, list(range(K)), return_dtype=pl.Int8).to_numpy().astype(np.int64)
def expected_f1(P, pred):
    tp = np.array([P[pred == k, k].sum() for k in range(K)])
    return float((2 * tp / (np.bincount(pred, minlength=K) + P.sum(0))).mean())

check = {}
for rel, real in LB_KNOWN.items():
    p = to_idx(BASE / rel)
    check[rel] = dict(simulasi=round(expected_f1(Ps, p), 4), nyata=real, selisih=round(expected_f1(Ps, p) - real, 4), cocok_dgn_kandidat=round(float((p == pred_s).mean()), 4))
print("cek skor LB yang diketahui:", json.dumps(check), flush=True)

sub = ids.with_columns(pl.Series("status_penanganan", [CLASSES[i] for i in pred_s.tolist()]))
sample = pl.read_csv(DATA / "sample_submission.csv", columns=["shipment_id"])
sub = sample.join(sub, on="shipment_id", how="left", maintain_order="left")
assert sub.height == sample.height and sub["status_penanganan"].null_count() == 0 and (sub["shipment_id"] == sample["shipment_id"]).all()
path = SUB / f"{OUT_NAME}.csv"
sub.write_csv(path)
np.save(RUN / "oof_blend_testlike.npy", Pm)
np.save(RUN / "oof_blend_full.npy", Pv)
np.save(RUN / "test_blend.npy", Ps)
report = dict(models=models, bobot=weights, logloss_validasi=round(ll(Pv, y), 5), logloss_meniru_test=round(ll(Pm, y), 5), macro_f1_crossfit_meniru_test=round(f1_cf, 5),
              f1_per_fold=per_fold, f1_per_kelas=dict(zip(CLASSES, np.round(per_class, 4).tolist())), pengali=dict(zip(CLASSES, np.round(w_final, 3).tolist())), cek_lb=check,
              sebaran_prediksi=dict(zip(CLASSES, np.round(np.bincount(pred_s, minlength=K) / len(pred_s), 4).tolist())), berkas=str(path), menit=round((time.time() - t0) / 60, 1))
(RUN / f"{OUT_NAME}_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
