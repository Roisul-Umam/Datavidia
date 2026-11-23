"""Uji kelompok fitur baru pada fold berkelompok yang sama (fold 0, latih 600 ribu baris)."""
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
from features_ext import build_ext

OUT = BASE / "exp" / "diagnostics"
tr, te, _ = build_all()
fx_tr, fx_te = build_ext()
assert (tr["shipment_id"] == fx_tr["shipment_id"]).all()
y = tr[TARGET].replace_strict(CLASSES, list(range(K)), return_dtype=pl.Int8).to_numpy().astype(np.int64)
fold = customer_folds(tr["customer_id"].rank("dense").to_numpy())
n_all, n_test = tr["cs_n"].to_numpy(), te["cs_n"].to_numpy()

drop = set(ID_COLS + NOISE_COLS + [TARGET, "hub_month", "courier_month"])
base = [c for c in te.columns if c not in drop]
hist = [c for c in fx_tr.columns if c.startswith("hist_")]
cx = [c for c in fx_tr.columns if c.startswith("cx_")]
load = [c for c in fx_tr.columns if c.startswith("load_")]
cats = [c for c in CAT_COLS if c in base]
print(f"base={len(base)} hist={len(hist)} cx={len(cx)} load={len(load)}", flush=True)

va = np.where(fold == 0)[0]
trn = np.where(fold != 0)[0]
rng = np.random.RandomState(42)
sub = rng.choice(trn, 600_000, replace=False)
yt, yv = y[sub], y[va]
prior = np.bincount(y[trn], minlength=K) / len(trn)
inner = rng.randint(0, 5, len(trn))
pos = np.full(len(y), -1, dtype=np.int64)
pos[trn] = np.arange(len(trn))
Tt, Tv, te_names = [], [], []
for key, alpha, p in [("courier_id", 10.0, "cr"), ("hub_id", 30.0, "hb"), ("hub_month", 15.0, "hm")]:
    codes = (tr[key].rank("dense").cast(pl.Int32) - 1).to_numpy()
    oof, C = te_oof(codes[trn], y[trn], int(codes.max()) + 1, prior, alpha, inner)
    Tt.append(oof[pos[sub]])
    Tv.append(te_apply(C, codes[va], prior, alpha))
    te_names += [f"te_{p}_{c[:6]}" for c in CLASSES] + [f"te_{p}_n"]
Tt, Tv = np.hstack(Tt), np.hstack(Tv)

Xb = tr.select(pl.col(base).cast(pl.Float32)).to_numpy()
Xh = fx_tr.select(pl.col(hist).cast(pl.Float32)).to_numpy()
Xc = fx_tr.select(pl.col(cx).cast(pl.Float32)).to_numpy()
Xl = fx_tr.select(pl.col(load).cast(pl.Float32)).to_numpy()
del tr, te, fx_tr, fx_te
gc.collect()

params = dict(objective="multiclass", num_class=K, metric="multi_logloss", learning_rate=0.08, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, max_bin=127, num_threads=14, verbose=-1, seed=42)
wv = test_like_weights(n_all[va], n_test)
res = {}
configs = {
    "A base": [(Xb, base)],
    "B base+hist": [(Xb, base), (Xh, hist)],
    "C base+hist+load": [(Xb, base), (Xh, hist), (Xl, load)],
    "D base+hist+cx": [(Xb, base), (Xh, hist), (Xc, cx)],
}
for name, parts in configs.items():
    t0 = time.time()
    Xt = np.hstack([p[sub] for p, _ in parts] + [Tt])
    Xv = np.hstack([p[va] for p, _ in parts] + [Tv])
    names = sum((n for _, n in parts), []) + te_names
    dt = lgb.Dataset(Xt, yt, feature_name=names, categorical_feature=cats, free_raw_data=True)
    dv = lgb.Dataset(Xv, yv, reference=dt, free_raw_data=True)
    m = lgb.train(params, dt, 1500, valid_sets=[dv], callbacks=[lgb.early_stopping(40, verbose=False)])
    P = m.predict(Xv, num_iteration=m.best_iteration)
    (f_cf, per), _ = crossfit_f1(P, yv)
    (f_w, _), _ = crossfit_f1(P, yv, wv)
    res[name] = dict(iters=m.best_iteration, logloss=round(log_loss(yv, P, labels=list(range(K))), 5), f1_argmax=round(f1_from_pred(yv, P.argmax(1))[0], 5),
                     f1_crossfit=round(f_cf, 5), f1_mirip_test=round(f_w, 5), per_kelas=dict(zip([c[:6] for c in CLASSES], np.round(per, 3).tolist())), menit=round((time.time() - t0) / 60, 1))
    print(name, json.dumps(res[name]), flush=True)
    gain = m.feature_importance("gain")
    groups = {"hist_": 0.0, "cx_": 0.0, "load_": 0.0, "cs_": 0.0, "te_": 0.0}
    for n, g in zip(names, gain):
        for pf in groups:
            if n.startswith(pf):
                groups[pf] += g
    print("   porsi gain:", {k: round(100 * v / gain.sum(), 1) for k, v in groups.items()}, "| teratas:", [names[i] for i in np.argsort(gain)[::-1][:14]], flush=True)
    del dt, dv, m, Xt, Xv, P
    gc.collect()
(OUT / "09_feature_groups.json").write_text(json.dumps(res, indent=2))
