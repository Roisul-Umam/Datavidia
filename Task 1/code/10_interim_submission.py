"""Kandidat sementara untuk menguji temuan agregat per split di leaderboard.

LightGBM dilatih pada fold 1-4 (berkelompok per pelanggan), fold 0 dipakai untuk early stopping
dan pengali kelas dengan bobot yang meniru sebaran jumlah saudara di test.
Berkas submission lama tidak disentuh.
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
from common import apply_weights, crossfit_f1, f1_from_pred, optimize_class_weights, te_apply, te_oof, test_like_weights
from features import BASE, CAT_COLS, CLASSES, DATA, ID_COLS, K, NOISE_COLS, TARGET, build_all, customer_folds
from features_ext import build_ext

OUT = BASE / "exp" / "submissions"
PRED = BASE / "exp" / "preds"
OUT.mkdir(parents=True, exist_ok=True)
PRED.mkdir(parents=True, exist_ok=True)
t0 = time.time()
tr, te, _ = build_all()
fx_tr, fx_te = build_ext()
assert (tr["shipment_id"] == fx_tr["shipment_id"]).all() and (te["shipment_id"] == fx_te["shipment_id"]).all()
y = tr[TARGET].replace_strict(CLASSES, list(range(K)), return_dtype=pl.Int8).to_numpy().astype(np.int64)
fold = customer_folds(tr["customer_id"].rank("dense").to_numpy())
n_all, n_test = tr["cs_n"].to_numpy(), te["cs_n"].to_numpy()
drop = set(ID_COLS + NOISE_COLS + [TARGET, "hub_month", "courier_month"])
base = [c for c in te.columns if c not in drop]
ext = [c for c in fx_tr.columns if c.startswith(("hist_", "cx_"))]
cats = [c for c in CAT_COLS if c in base]
names_f = base + ext

va = np.where(fold == 0)[0]
trn = np.where(fold != 0)[0]
rng = np.random.RandomState(42)
prior = np.bincount(y[trn], minlength=K) / len(trn)
inner = rng.randint(0, 5, len(trn))
Tt, Tv, Ts, te_names = [], [], [], []
for key, alpha, p in [("courier_id", 10.0, "cr"), ("hub_id", 30.0, "hb"), ("hub_month", 15.0, "hm")]:
    both = pl.concat([tr.select(key), te.select(key)])[key].rank("dense").cast(pl.Int32) - 1
    codes, codes_te = both[:tr.height].to_numpy(), both[tr.height:].to_numpy()
    nc = int(both.max()) + 1
    oof, C = te_oof(codes[trn], y[trn], nc, prior, alpha, inner)
    Tt.append(oof)
    Tv.append(te_apply(C, codes[va], prior, alpha))
    Ts.append(te_apply(C, codes_te, prior, alpha))
    te_names += [f"te_{p}_{c[:6]}" for c in CLASSES] + [f"te_{p}_n"]
Tt, Tv, Ts = np.hstack(Tt), np.hstack(Tv), np.hstack(Ts)
names = names_f + te_names


def matrix(df, fx):
    return np.hstack([df.select(pl.col(base).cast(pl.Float32)).to_numpy(), fx.select(pl.col(ext).cast(pl.Float32)).to_numpy()])


Xa = matrix(tr, fx_tr)
Xs = np.hstack([matrix(te, fx_te), Ts])
Xt = np.hstack([Xa[trn], Tt])
Xv = np.hstack([Xa[va], Tv])
ids_te = te["shipment_id"]
del Xa, Tt, Tv, Ts, tr, te, fx_tr, fx_te
gc.collect()
print(f"matriks latih {Xt.shape}, validasi {Xv.shape}, test {Xs.shape} ({time.time() - t0:.0f}s)", flush=True)

params = dict(objective="multiclass", num_class=K, metric="multi_logloss", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, max_bin=127, num_threads=14, verbose=-1, seed=42)
dt = lgb.Dataset(Xt, y[trn], feature_name=names, categorical_feature=cats, free_raw_data=True)
dv = lgb.Dataset(Xv, y[va], reference=dt, free_raw_data=True)
m = lgb.train(params, dt, 2000, valid_sets=[dv], callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])
Pv = m.predict(Xv, num_iteration=m.best_iteration)
Ps = m.predict(Xs, num_iteration=m.best_iteration)
yv = y[va]
wv = test_like_weights(n_all[va], n_test)
(f_cf, per), _ = crossfit_f1(Pv, yv)
(f_w, per_w), _ = crossfit_f1(Pv, yv, wv)
w_final, _ = optimize_class_weights(Pv, yv, wv, passes=3)
pred = apply_weights(Ps, w_final)
sub = pl.DataFrame({"shipment_id": ids_te, "status_penanganan": [CLASSES[i] for i in pred.tolist()]})
sample = pl.read_csv(DATA.parent / "datasets" / "sample_submission.csv", columns=["shipment_id"])
sub = sample.join(sub, on="shipment_id", how="left")
assert sub.height == sample.height and sub["status_penanganan"].null_count() == 0
path = OUT / "sub_interim_lgbm.csv"
sub.write_csv(path)
np.save(PRED / "interim_val_proba.npy", Pv.astype(np.float32))
np.save(PRED / "interim_test_proba.npy", Ps.astype(np.float32))
report = dict(iters=m.best_iteration, logloss=round(log_loss(yv, Pv, labels=list(range(K))), 5), f1_argmax=round(f1_from_pred(yv, Pv.argmax(1))[0], 5),
              f1_crossfit=round(f_cf, 5), f1_mirip_test=round(f_w, 5), f1_kelas_mirip_test=dict(zip(CLASSES, np.round(per_w, 4).tolist())),
              pengali=dict(zip(CLASSES, np.round(w_final, 3).tolist())),
              sebaran_prediksi_test=dict(zip(CLASSES, np.round(np.bincount(pred, minlength=K) / len(pred), 4).tolist())), menit=round((time.time() - t0) / 60, 1))
(OUT / "sub_interim_lgbm.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
print("berkas:", path, flush=True)
