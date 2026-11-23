"""Model penggerak riwayat, diuji dengan label nyata dan dengan simulasi leaderboard.

C1  fold berkelompok per pelanggan, tanpa target encoding pelanggan
C2  C1 + sebagian baris latih dikosongkan riwayatnya (meniru 11,4% baris test tanpa riwayat)
C3  C2 pada fold acak per baris + target encoding pelanggan dari baris saudara
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
from features import BASE, CAT_COLS, CLASSES, ID_COLS, K, NOISE_COLS, TARGET, build_all, customer_folds
from features_driver import build_driver

OUT, PRED = BASE / "exp" / "diagnostics", BASE / "exp" / "preds"
NO_HISTORY = 0.1142
t0 = time.time()
tr, te, _ = build_all()
d_tr, d_te, info = build_driver()
y = tr[TARGET].replace_strict(CLASSES, list(range(K)), return_dtype=pl.Int8).to_numpy().astype(np.int64)
group_fold = customer_folds(tr["customer_id"].rank("dense").to_numpy())
row_fold = np.random.RandomState(2024).randint(0, 5, len(y))
drop = set(ID_COLS + NOISE_COLS + [TARGET, "hub_month", "courier_month"]) | set(info["own_drop"])
base = [c for c in te.columns if c not in drop and not c.startswith("cs_")]
drv = [c for c in d_tr.columns if c.startswith("drv_")]
cats = [c for c in CAT_COLS if c in base]
print(f"fitur baris={len(base)} penggerak={len(drv)} kategori={len(cats)}", flush=True)
Xa = np.hstack([tr.select(pl.col(base).cast(pl.Float32)).to_numpy(), d_tr.select(pl.col(drv).cast(pl.Float32)).to_numpy()])
Xs = np.hstack([te.select(pl.col(base).cast(pl.Float32)).to_numpy(), d_te.select(pl.col(drv).cast(pl.Float32)).to_numpy()])
names0 = base + drv
drv_idx = np.arange(len(base), len(names0))
codes = {}
for key in ("courier_id", "hub_id", "hub_month", "customer_id"):
    both = pl.concat([tr.select(key), te.select(key)])[key].rank("dense").cast(pl.Int32) - 1
    codes[key] = (both[:tr.height].to_numpy(), both[tr.height:].to_numpy(), int(both.max()) + 1)
ship_tr, cust_tr = tr["shipment_id"], tr["customer_id"]
del tr, te, d_tr, d_te
gc.collect()

hyp = np.load(PRED / "hypothesis_test_proba.npz")
P_h = hyp["train_saja_nm_1"].astype(np.float32) / 65535
sub_ids = pl.read_parquet(BASE / "exp" / "cache" / "fe_test.parquet", columns=["shipment_id"])


def to_idx(path):
    d = sub_ids.join(pl.read_csv(path), on="shipment_id", how="left", maintain_order="left")
    return d["status_penanganan"].replace_strict(CLASSES, list(range(K)), return_dtype=pl.Int8).to_numpy().astype(np.int64)


pred_old, pred_int = to_idx(BASE / "submission.csv"), to_idx(BASE / "exp" / "submissions" / "sub_interim_lgbm.csv")


def expected_f1(P, pred):
    tp = np.array([P[pred == k, k].sum() for k in range(K)])
    return float((2 * tp / (np.bincount(pred, minlength=K) + P.sum(0))).mean())


def encode(a, b, keys, seed=7):
    rng = np.random.RandomState(seed)
    prior = np.bincount(y[a], minlength=K) / len(a)
    inner = rng.randint(0, 5, len(a))
    Ta, Tb, Tc, nm = [], [], [], []
    for key, alpha, p in keys:
        cd, cd_te, nc = codes[key]
        oof, C = te_oof(cd[a], y[a], nc, prior, alpha, inner)
        Ta.append(oof)
        Tb.append(te_apply(C, cd[b], prior, alpha))
        Tc.append(te_apply(C, cd_te, prior, alpha))
        nm += [f"te_{p}_{c[:6]}" for c in CLASSES] + [f"te_{p}_n"]
    return np.hstack(Ta), np.hstack(Tb), np.hstack(Tc), nm


params = dict(objective="multiclass", num_class=K, metric="multi_logloss", learning_rate=0.06, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, max_bin=127, num_threads=14, verbose=-1, seed=42)
ENT = [("courier_id", 10.0, "cr"), ("hub_id", 30.0, "hb"), ("hub_month", 15.0, "hm")]
results = {}


def run(name, fold_vec, mask_train, keys):
    t1 = time.time()
    a, b = np.where(fold_vec != 0)[0], np.where(fold_vec == 0)[0]
    Ta, Tb, Tc, nm = encode(a, b, keys)
    Xt = np.hstack([Xa[a], Ta])
    Xv = np.hstack([Xa[b], Tb])
    if mask_train:
        rng = np.random.RandomState(11)
        k = rng.random(len(a)) < NO_HISTORY
        Xt[np.ix_(np.where(k)[0], drv_idx)] = np.nan
        Xt[k, names0.index("drv_n")] = 0
    names = names0 + nm
    dt = lgb.Dataset(Xt, y[a], feature_name=names, categorical_feature=cats, free_raw_data=True)
    dv = lgb.Dataset(Xv, y[b], reference=dt, free_raw_data=True)
    m = lgb.train(params, dt, 1500, valid_sets=[dv], callbacks=[lgb.early_stopping(40, verbose=False)])
    Pv = m.predict(Xv, num_iteration=m.best_iteration)
    Xv_m = Xv.copy()
    k = np.random.RandomState(12).random(len(b)) < NO_HISTORY
    Xv_m[np.ix_(np.where(k)[0], drv_idx)] = np.nan
    Xv_m[k, names0.index("drv_n")] = 0
    Pm = m.predict(Xv_m, num_iteration=m.best_iteration)
    Ps = m.predict(np.hstack([Xs, Tc]), num_iteration=m.best_iteration)
    (f_cf, per), _ = crossfit_f1(Pv, y[b])
    (f_mask, _), _ = crossfit_f1(Pm, y[b])
    w, _ = optimize_class_weights(Pm, y[b], passes=3)
    pred_s = apply_weights(Ps, w)
    gain = m.feature_importance("gain")
    groups = {"drv_": 0.0, "te_": 0.0, "lain": 0.0}
    for n, g_ in zip(names, gain):
        groups["drv_" if n.startswith("drv_") else "te_" if n.startswith("te_") else "lain"] += g_
    r = dict(iters=m.best_iteration, logloss=round(log_loss(y[b], Pv, labels=list(range(K))), 5), f1_argmax=round(f1_from_pred(y[b], Pv.argmax(1))[0], 5),
             f1_crossfit=round(f_cf, 5), f1_riwayat_kosong_11pct=round(f_mask, 5), per_kelas=dict(zip([c[:6] for c in CLASSES], np.round(per, 3).tolist())),
             lb_harapan_kandidat=round(expected_f1(P_h, pred_s), 5), cocok_dgn_lama=round(float((pred_s == pred_old).mean()), 4), cocok_dgn_sementara=round(float((pred_s == pred_int).mean()), 4),
             pengali=np.round(w, 3).tolist(), porsi_gain={k_: round(100 * v / gain.sum(), 1) for k_, v in groups.items()}, menit=round((time.time() - t1) / 60, 1))
    results[name] = r
    print(name, json.dumps(r), flush=True)
    print("   teratas:", [names[i] for i in np.argsort(gain)[::-1][:22]], flush=True)
    np.save(PRED / f"drv_{name}_test.npy", Ps.astype(np.float32))
    del dt, dv, m, Xt, Xv, Xv_m, Ta, Tb, Tc, Pv, Pm, Ps
    gc.collect()


print(f"acuan LB harapan: lama={expected_f1(P_h, pred_old):.4f} sementara={expected_f1(P_h, pred_int):.4f}", flush=True)
run("C1", group_fold, False, ENT)
run("C2", group_fold, True, ENT)
run("C3", row_fold, True, ENT + [("customer_id", 3.0, "cu")])
(OUT / "14_driver_model.json").write_text(json.dumps(results, indent=2))
print(f"selesai ({(time.time() - t0) / 60:.1f} menit)", flush=True)
