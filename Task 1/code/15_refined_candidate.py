"""Kandidat terperbaiki: riwayat dari train + target encoding pelanggan.

Perbaikan dari konfigurasi C3:
  * Baris tanpa riwayat (latih dan validasi) dikosongkan penggerak DAN encoding pelanggannya, sama seperti baris test.
  * Target encoding memakai 10 fold dalam.
  * Encoding untuk test dihitung dari seluruh label train.
Sebelum menulis berkas, kalibrasi metode diperiksa pada label nyata dan kedua skor LB yang diketahui.
"""
import gc
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import log_loss

sys.path.insert(0, r"E:\Kuliyah\Datavidia\Task 1\code")
from common import apply_weights, crossfit_f1, f1_from_pred, optimize_class_weights, te_apply, te_oof
from features import BASE, CAT_COLS, CLASSES, DATA, ID_COLS, K, NOISE_COLS, TARGET, build_all
from features_driver import build_driver

OUT, PRED, SUB = BASE / "exp" / "diagnostics", BASE / "exp" / "preds", BASE / "exp" / "submissions"
LB_OLD, LB_INT = 0.59364, 0.48574
NO_HISTORY = 0.1142
t0 = time.time()
tr, te, _ = build_all()
d_tr, d_te, info = build_driver()
y = tr[TARGET].replace_strict(CLASSES, list(range(K)), return_dtype=pl.Int8).to_numpy().astype(np.int64)
row_fold = np.random.RandomState(2024).randint(0, 5, len(y))
drop = set(ID_COLS + NOISE_COLS + [TARGET, "hub_month", "courier_month"]) | set(info["own_drop"])
base = [c for c in te.columns if c not in drop and not c.startswith("cs_")]
drv = [c for c in d_tr.columns if c.startswith("drv_")]
cats = [c for c in CAT_COLS if c in base]
Xa = np.hstack([tr.select(pl.col(base).cast(pl.Float32)).to_numpy(), d_tr.select(pl.col(drv).cast(pl.Float32)).to_numpy()])
Xs = np.hstack([te.select(pl.col(base).cast(pl.Float32)).to_numpy(), d_te.select(pl.col(drv).cast(pl.Float32)).to_numpy()])
names0 = base + drv
drv_idx = np.arange(len(base), len(names0))
ENT = [("courier_id", 10.0, "cr"), ("hub_id", 30.0, "hb"), ("hub_month", 15.0, "hm"), ("customer_id", 3.0, "cu")]
codes = {}
for key, _, _ in ENT:
    both = pl.concat([tr.select(key), te.select(key)])[key].rank("dense").cast(pl.Int32) - 1
    codes[key] = (both[:tr.height].to_numpy(), both[tr.height:].to_numpy(), int(both.max()) + 1)
ids_te = te["shipment_id"]
no_hist_test = d_te["drv_n"].to_numpy() == 0
print(f"baris test tanpa riwayat: {no_hist_test.mean():.4f}", flush=True)
del tr, te, d_tr, d_te
gc.collect()

a, b = np.where(row_fold != 0)[0], np.where(row_fold == 0)[0]
prior_a, prior_all = np.bincount(y[a], minlength=K) / len(a), np.bincount(y, minlength=K) / len(y)
inner = np.random.RandomState(7).randint(0, 10, len(a))
Ta, Tb, Tc, nm = [], [], [], []
for key, alpha, p in ENT:
    cd, cd_te, nc = codes[key]
    oof, C = te_oof(cd[a], y[a], nc, prior_a, alpha, inner)
    Ta.append(oof)
    Tb.append(te_apply(C, cd[b], prior_a, alpha))
    C_all = np.zeros((nc, K), dtype=np.int32)
    np.add.at(C_all, (cd, y), 1)
    Tc.append(te_apply(C_all, cd_te, prior_all, alpha))
    nm += [f"te_{p}_{c[:6]}" for c in CLASSES] + [f"te_{p}_n"]
Ta, Tb, Tc = np.hstack(Ta), np.hstack(Tb), np.hstack(Tc)
names = names0 + nm
cu_idx = np.array([names.index(f"te_cu_{c[:6]}") for c in CLASSES])
cu_n, drv_n = names.index("te_cu_n"), names.index("drv_n")


def blank_history(X, rows, prior):
    X[np.ix_(rows, drv_idx)] = np.nan
    X[rows, drv_n] = 0
    X[np.ix_(rows, cu_idx)] = prior[None, :]
    X[rows, cu_n] = 0


Xt, Xv = np.hstack([Xa[a], Ta]), np.hstack([Xa[b], Tb])
Xtest = np.hstack([Xs, Tc])
del Xa, Xs, Ta, Tb, Tc
gc.collect()
blank_history(Xt, np.where(np.random.RandomState(11).random(len(a)) < NO_HISTORY)[0], prior_a)
Xtest[np.ix_(np.where(no_hist_test)[0], cu_idx)] = prior_all[None, :]
Xtest[no_hist_test, cu_n] = 0
print(f"matriks {Xt.shape} ({time.time() - t0:.0f}s)", flush=True)

params = dict(objective="multiclass", num_class=K, metric="multi_logloss", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, max_bin=127, num_threads=14, verbose=-1, seed=42)
dt = lgb.Dataset(Xt, y[a], feature_name=names, categorical_feature=cats, free_raw_data=True)
dv = lgb.Dataset(Xv, y[b], reference=dt, free_raw_data=True)
m = lgb.train(params, dt, 2000, valid_sets=[dv], callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])
m.save_model(str(BASE / "exp" / "models" / "r0_lgbm.txt"), num_iteration=m.best_iteration)
Pv = m.predict(Xv, num_iteration=m.best_iteration)
Xm = Xv.copy()
blank_history(Xm, np.where(np.random.RandomState(12).random(len(b)) < NO_HISTORY)[0], prior_a)
Pm = m.predict(Xm, num_iteration=m.best_iteration)
Ps = m.predict(Xtest, num_iteration=m.best_iteration)
del Xm, Xt, Xv, Xtest, dt, dv
gc.collect()


def expected_f1(P, pred):
    tp = np.array([P[pred == k, k].sum() for k in range(K)])
    return float((2 * tp / (np.bincount(pred, minlength=K) + P.sum(0))).mean())


yv = y[b]
(f_cf, per), _ = crossfit_f1(Pv, yv)
(f_mask, per_m), _ = crossfit_f1(Pm, yv)
w, _ = optimize_class_weights(Pm, yv, passes=3)
pred_s = apply_weights(Ps, w)

ids = pl.DataFrame({"shipment_id": ids_te})
def to_idx(path):
    d = ids.join(pl.read_csv(path), on="shipment_id", how="left", maintain_order="left")
    return d["status_penanganan"].replace_strict(CLASSES, list(range(K)), return_dtype=pl.Int8).to_numpy().astype(np.int64)
pred_old, pred_int = to_idx(BASE / "submission.csv"), to_idx(SUB / "sub_interim_lgbm.csv")
oof_old = np.load(BASE / "models" / "oof_predictions.npz")["pred_crossfit"].astype(np.int64)[b]

sub = ids.with_columns(pl.Series("status_penanganan", [CLASSES[i] for i in pred_s.tolist()]))
sample = pl.read_csv(DATA / "sample_submission.csv", columns=["shipment_id"])
sub = sample.join(sub, on="shipment_id", how="left", maintain_order="left")
assert sub.height == sample.height and sub["status_penanganan"].null_count() == 0 and (sub["shipment_id"] == sample["shipment_id"]).all()
path = SUB / "sub_r0_history_te.csv"
sub.write_csv(path)
np.save(PRED / "r0_val_proba.npy", Pv.astype(np.float32))
np.save(PRED / "r0_val_blank_proba.npy", Pm.astype(np.float32))
np.save(PRED / "r0_test_proba.npy", Ps.astype(np.float32))
np.save(PRED / "r0_val_idx.npy", b)
report = dict(
    iters=m.best_iteration, logloss=round(log_loss(yv, Pv, labels=list(range(K))), 5),
    f1_validasi=round(f_cf, 5), f1_validasi_11pct_tanpa_riwayat=round(f_mask, 5),
    f1_kelas=dict(zip(CLASSES, np.round(per_m, 4).tolist())), pengali=dict(zip(CLASSES, np.round(w, 3).tolist())),
    kalibrasi_val_model_lama=dict(nyata=round(f1_from_pred(yv, oof_old)[0], 4), harapan=round(expected_f1(Pv, oof_old), 4)),
    cek_lb=dict(lama_harapan=round(expected_f1(Ps, pred_old), 4), lama_nyata=LB_OLD, sementara_harapan=round(expected_f1(Ps, pred_int), 4), sementara_nyata=LB_INT),
    lb_harapan_kandidat=round(expected_f1(Ps, pred_s), 4), cocok_dgn_lama=round(float((pred_s == pred_old).mean()), 4),
    sebaran_prediksi=dict(zip(CLASSES, np.round(np.bincount(pred_s, minlength=K) / len(pred_s), 4).tolist())), menit=round((time.time() - t0) / 60, 1))
(SUB / "sub_r0_history_te.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
print("berkas:", path, flush=True)
