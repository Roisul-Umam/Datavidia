"""Uji cepat pada satu fold berkelompok per pelanggan: baris saja vs baris + agregat pelanggan per split."""
import gc
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import log_loss

sys.path.insert(0, r"E:\Kuliyah\Datavidia\Task 1\code")
from common import crossfit_f1, f1_from_pred, te_apply, te_oof, test_like_weights
from features import BASE, CAT_COLS, CLASSES, ID_COLS, K, NOISE_COLS, TARGET, build_all, customer_folds

OUT = BASE / "exp" / "diagnostics"
tr, te, meta = build_all()
y = tr[TARGET].replace_strict(CLASSES, list(range(K)), return_dtype=pl.Int8).to_numpy().astype(np.int64)
fold = customer_folds(tr["customer_id"].rank("dense").to_numpy())
n_val_all = tr["cs_n"].to_numpy()
n_test = te["cs_n"].to_numpy()
print("sebaran cs_n (baris) train:", np.round(np.bincount(np.minimum(n_val_all.astype(int), 6), minlength=7)[1:] / len(n_val_all), 3))
print("sebaran cs_n (baris) test :", np.round(np.bincount(np.minimum(n_test.astype(int), 6), minlength=7)[1:] / len(n_test), 3), flush=True)

drop = set(ID_COLS + NOISE_COLS + [TARGET, "hub_month", "courier_month"])
all_cols = [c for c in te.columns if c not in drop]
cs_cols = [c for c in all_cols if c.startswith("cs_")]
row_cols = [c for c in all_cols if not c.startswith("cs_")]
cats = [c for c in CAT_COLS if c in all_cols]
print(f"fitur baris={len(row_cols)} | fitur pelanggan per split={len(cs_cols)}", flush=True)

va = np.where(fold == 0)[0]
trn = np.where(fold != 0)[0]
rng = np.random.RandomState(42)
sub = rng.choice(trn, min(600_000, len(trn)), replace=False)
yt, yv = y[sub], y[va]
prior = np.bincount(y[trn], minlength=K) / len(trn)
inner = rng.randint(0, 5, len(trn))
pos = np.full(len(y), -1, dtype=np.int64)
pos[trn] = np.arange(len(trn))

Tt, Tv = [], []
te_names = []
for key, alpha, p in [("courier_id", 10.0, "cr"), ("hub_id", 30.0, "hb"), ("hub_month", 15.0, "hm")]:
    codes = (tr[key].rank("dense").cast(pl.Int32) - 1).to_numpy()
    oof, C = te_oof(codes[trn], y[trn], int(codes.max()) + 1, prior, alpha, inner)
    Tt.append(oof[pos[sub]])
    Tv.append(te_apply(C, codes[va], prior, alpha))
    te_names += [f"te_{p}_{c[:6]}" for c in CLASSES] + [f"te_{p}_n"]
Tt, Tv = np.hstack(Tt), np.hstack(Tv)
print("target encoding kurir/hub selesai", flush=True)

params = dict(objective="multiclass", num_class=K, metric="multi_logloss", learning_rate=0.08, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, max_bin=127, num_threads=14, verbose=-1, seed=42)

res = {}
wv = test_like_weights(n_val_all[va], n_test)
for name, cols in [("baris", row_cols), ("baris+pelanggan", row_cols + cs_cols)]:
    t0 = time.time()
    Xt = np.hstack([tr[sub.tolist()].select(pl.col(cols).cast(pl.Float32)).to_numpy(), Tt]) if False else None
    Xall = tr.select(pl.col(cols).cast(pl.Float32)).to_numpy()
    Xt = np.hstack([Xall[sub], Tt])
    Xv = np.hstack([Xall[va], Tv])
    del Xall
    gc.collect()
    names = cols + te_names
    catf = [c for c in cats if c in cols]
    dt = lgb.Dataset(Xt, yt, feature_name=names, categorical_feature=catf, free_raw_data=True)
    dv = lgb.Dataset(Xv, yv, reference=dt, free_raw_data=True)
    m = lgb.train(params, dt, 1500, valid_sets=[dv], callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(100)])
    P = m.predict(Xv, num_iteration=m.best_iteration)
    ll = log_loss(yv, P, labels=list(range(K)))
    f_arg, per_arg = f1_from_pred(yv, P.argmax(1))
    (f_cf, per_cf), _ = crossfit_f1(P, yv)
    (f_w, per_w), _ = crossfit_f1(P, yv, wv)
    res[name] = dict(iters=m.best_iteration, logloss=round(ll, 5), f1_argmax=round(f_arg, 5), f1_crossfit=round(f_cf, 5), f1_mirip_test=round(f_w, 5),
                     per_kelas=dict(zip([c[:6] for c in CLASSES], np.round(per_cf, 3).tolist())), menit=round((time.time() - t0) / 60, 1))
    print(name, json.dumps(res[name]), flush=True)
    if name == "baris+pelanggan":
        gain = m.feature_importance("gain")
        top = np.argsort(gain)[::-1][:40]
        print("fitur teratas:", [(names[i], round(100 * gain[i] / gain.sum(), 1)) for i in top], flush=True)
        for b in range(1, 7):
            k = np.minimum(n_val_all[va].astype(int), 6) == b
            print(f"  cs_n={b}: baris={k.sum():>6d}  F1 argmax={f1_from_pred(yv[k], P[k].argmax(1))[0]:.4f}", flush=True)
    del dt, dv, m, Xt, Xv, P
    gc.collect()
(OUT / "05_quick_cv.json").write_text(json.dumps(res, indent=2))
