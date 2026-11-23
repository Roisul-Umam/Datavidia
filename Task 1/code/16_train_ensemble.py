"""Ensemble 5-fold: LightGBM, XGBoost, CatBoost, dan regresi logistik multinomial.

Rancangan fitur mengikuti mekanisme yang ditemukan:
  * Sinyal milik baris sendiri tetap dipakai dari baris itu.
  * Sinyal riwayat (drv_*) untuk baris test dihitung dari baris TRAIN pelanggan yang sama.
  * Sebagian baris latih dan validasi dikosongkan riwayatnya, setara porsi pelanggan test tanpa riwayat.
  * Target encoding kurir, hub, hub-bulan, dan pelanggan memakai 10 fold dalam.

Setiap kombinasi fold dan model punya checkpoint sendiri, jadi proses bisa dihentikan dan dilanjutkan.
Tidak menyentuh folder models/ maupun submission.csv.
Jalankan uji cepat dengan variabel lingkungan SMOKE=1.
"""
import gc
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, r"E:\Kuliyah\Datavidia\Task 1\code")
from common import te_apply, te_oof
from features import BASE, CAT_COLS, CLASSES, ID_COLS, K, NOISE_COLS, TARGET, build_all
from features_driver import build_driver

SMOKE = os.environ.get("SMOKE", "0") == "1"
RUN = BASE / "exp" / ("v2_smoke" if SMOKE else "v2")
MODEL_DIR, PRED_DIR = RUN / "models", RUN / "preds"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)
SEED, THREADS, N_FOLDS, TE_INNER = 42, 14, 5, 10
MODELS = ["lgbm", "xgb", "cat", "glm"]
ROUNDS = dict(lgbm=30, xgb=20, cat=30) if SMOKE else dict(lgbm=3000, xgb=2500, cat=4000)
STOP = 10 if SMOKE else 70
VERB = 10 if SMOKE else 200
ENT = [("courier_id", 10.0, "cr"), ("hub_id", 30.0, "hb"), ("hub_month", 15.0, "hm"), ("customer_id", 3.0, "cu")]
LABELS = list(range(K))
T0 = time.time()


def log(msg):
    print(f"[{(time.time() - T0) / 60:6.1f}m] {msg}", flush=True)


tr, te, _ = build_all()
d_tr, d_te, info = build_driver()
y = tr[TARGET].replace_strict(CLASSES, LABELS, return_dtype=pl.Int8).to_numpy().astype(np.int64)
NO_HISTORY = float((d_te["drv_n"] == 0).mean())

booster = lgb.Booster(model_file=str(BASE / "exp" / "models" / "r0_lgbm.txt"))
gain = booster.feature_importance("gain").astype(float)
share = dict(zip(booster.feature_name(), gain / gain.sum()))
KEEP = {n for n, s in share.items() if s >= 1e-4} | {"drv_n"}
del booster

drop = set(ID_COLS + NOISE_COLS + [TARGET, "hub_month", "courier_month"]) | set(info["own_drop"])
base = [c for c in te.columns if c not in drop and not c.startswith("cs_") and c in KEEP]
drv = [c for c in d_tr.columns if c.startswith("drv_") and c in KEEP]
Xa = np.hstack([tr.select(pl.col(base).cast(pl.Float32)).to_numpy(), d_tr.select(pl.col(drv).cast(pl.Float32)).to_numpy()])
Xs = np.hstack([te.select(pl.col(base).cast(pl.Float32)).to_numpy(), d_te.select(pl.col(drv).cast(pl.Float32)).to_numpy()])
te_all = [f"te_{p}_{c[:6]}" for _, _, p in ENT for c in CLASSES + ["n_____"]]
te_all = [n.replace("_n_____", "_n") for n in te_all]
te_keep = [i for i, n in enumerate(te_all) if n in KEEP]
cols = base + drv + [te_all[i] for i in te_keep]
ix = {n: i for i, n in enumerate(cols)}
cats = [c for c in CAT_COLS if c in ix]
drv_ix = [ix[c] for c in drv if c != "drv_n"]
cu_ix = [(ix[f"te_cu_{c[:6]}"], k) for k, c in enumerate(CLASSES) if f"te_cu_{c[:6]}" in ix]
codes = {}
for key, _, _ in ENT:
    both = pl.concat([tr.select(key), te.select(key)])[key].rank("dense").cast(pl.Int32) - 1
    codes[key] = (both[:tr.height].to_numpy(), both[tr.height:].to_numpy(), int(both.max()) + 1)
ids_te = te["shipment_id"].to_list()
del tr, te, d_tr, d_te
gc.collect()

prior_all = np.bincount(y, minlength=K) / len(y)
Ts = []
for key, alpha, _ in ENT:
    cd, cd_te, nc = codes[key]
    C_all = np.zeros((nc, K), dtype=np.int32)
    np.add.at(C_all, (cd, y), 1)
    Ts.append(te_apply(C_all, cd_te, prior_all, alpha))
    del C_all
Xtest = np.hstack([Xs, np.hstack(Ts)[:, te_keep]]).astype(np.float32)
del Xs, Ts
gc.collect()
folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
fold_id = np.zeros(len(y), dtype=np.int8)
for f, (_, v) in enumerate(folds):
    fold_id[v] = f
np.save(RUN / "fold_id.npy", fold_id)
np.save(RUN / "y.npy", y.astype(np.int8))
(RUN / "config.json").write_text(json.dumps(dict(columns=cols, categorical=cats, n_folds=N_FOLDS, seed=SEED, no_history_share=NO_HISTORY, classes=CLASSES,
                                                 te_entities=[list(e) for e in ENT], te_inner=TE_INNER, shipment_order="sorted shipment_id"), ensure_ascii=False, indent=1))
if SMOKE:
    Xtest = Xtest[:20_000]
log(f"siap: {len(cols)} fitur ({len(base)} baris, {len(drv)} riwayat, {len(te_keep)} encoding) | tanpa riwayat {NO_HISTORY:.4f} | test {Xtest.shape}")


def blank(X, rows, prior):
    if len(rows) == 0:
        return
    X[np.ix_(rows, drv_ix)] = np.nan
    X[rows, ix["drv_n"]] = 0
    for i, k in cu_ix:
        X[rows, i] = prior[k]
    if "te_cu_n" in ix:
        X[rows, ix["te_cu_n"]] = 0


def fit_lgbm(f, Xt, yt, Xm, ym, Xv):
    params = dict(objective="multiclass", num_class=K, metric="multi_logloss", learning_rate=0.04, num_leaves=63, min_data_in_leaf=100 if SMOKE else 200,
                  feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, max_bin=255, num_threads=THREADS, verbose=-1, seed=SEED + f)
    dt = lgb.Dataset(Xt, yt, feature_name=cols, categorical_feature=cats, free_raw_data=True)
    dm = lgb.Dataset(Xm, ym, reference=dt, free_raw_data=True)
    m = lgb.train(params, dt, ROUNDS["lgbm"], valid_sets=[dm], callbacks=[lgb.early_stopping(STOP, verbose=False), lgb.log_evaluation(VERB)])
    it = m.best_iteration or ROUNDS["lgbm"]
    m.save_model(str(MODEL_DIR / f"lgbm_f{f}.txt"), num_iteration=it)
    out = (m.predict(Xv, num_iteration=it), m.predict(Xm, num_iteration=it), m.predict(Xtest, num_iteration=it))
    del dt, dm, m
    return out, it


def fit_xgb(f, Xt, yt, Xm, ym, Xv):
    params = dict(objective="multi:softprob", num_class=K, tree_method="hist", device="cuda", eta=0.05, max_depth=7, min_child_weight=50, subsample=0.8,
                  colsample_bytree=0.6, reg_lambda=5.0, max_bin=256, eval_metric="mlogloss", seed=SEED + f, nthread=THREADS)
    dt = xgb.DMatrix(Xt, label=yt, nthread=THREADS)
    dm = xgb.DMatrix(Xm, label=ym, nthread=THREADS)
    try:
        b = xgb.train(params, dt, ROUNDS["xgb"], evals=[(dm, "val")], early_stopping_rounds=STOP, verbose_eval=VERB)
    except xgb.core.XGBoostError as e:
        log(f"XGBoost GPU gagal, pindah ke CPU: {str(e)[:160]}")
        params["device"] = "cpu"
        b = xgb.train(params, dt, ROUNDS["xgb"], evals=[(dm, "val")], early_stopping_rounds=STOP, verbose_eval=VERB)
    it = b.best_iteration + 1
    b = b[:it]
    b.set_param({"device": "cpu", "nthread": THREADS})
    b.save_model(str(MODEL_DIR / f"xgb_f{f}.json"))
    out = (b.predict(xgb.DMatrix(Xv, nthread=THREADS)), b.predict(dm), b.predict(xgb.DMatrix(Xtest, nthread=THREADS)))
    del dt, dm, b
    return out, it


def fit_cat(f, Xt, yt, Xm, ym, Xv):
    pm = dict(loss_function="MultiClass", iterations=ROUNDS["cat"], learning_rate=0.06, depth=7, l2_leaf_reg=5.0, bootstrap_type="Bernoulli", subsample=0.8,
              border_count=128, od_type="Iter", od_wait=STOP, random_seed=SEED + f, verbose=VERB, allow_writing_files=False, thread_count=THREADS)
    try:
        m = CatBoostClassifier(**pm, task_type="GPU", devices="0", gpu_ram_part=0.85)
        m.fit(Xt, yt, eval_set=(Xm, ym), use_best_model=True)
    except Exception as e:
        log(f"CatBoost GPU gagal, pindah ke CPU 600 ribu baris: {str(e)[:200]}")
        sub = np.random.RandomState(SEED + f).choice(len(yt), min(600_000, len(yt)), replace=False)
        m = CatBoostClassifier(**pm, rsm=0.7)
        m.fit(Xt[sub], yt[sub], eval_set=(Xm, ym), use_best_model=True)
    it = int(m.tree_count_)
    m.save_model(str(MODEL_DIR / f"cat_f{f}.cbm"))
    out = (m.predict_proba(Xv), m.predict_proba(Xm), m.predict_proba(Xtest))
    del m
    return out, it


def glm_stats(X, seed):
    num = [i for i, c in enumerate(cols) if c not in cats]
    names = [cols[i] for i in num]
    tp = [j for j, c in enumerate(names) if c.startswith("te_") and not c.endswith("_n")]
    np_ = [j for j, c in enumerate(names) if c.startswith("te_") and c.endswith("_n")]
    s = X[np.random.RandomState(seed).choice(len(X), min(200_000, len(X)), replace=False)][:, num].copy()
    s[:, tp] = np.log(np.clip(s[:, tp], 1e-4, 1))
    s[:, np_] = np.log1p(s[:, np_])
    lo, med, hi = np.empty(s.shape[1]), np.empty(s.shape[1]), np.empty(s.shape[1])
    for j in range(s.shape[1]):
        v = s[:, j]
        v = v[~np.isnan(v)]
        lo[j], med[j], hi[j] = np.percentile(v, [0.5, 50, 99.5]) if len(v) else (0, 0, 0)
    c = np.clip(np.where(np.isnan(s), med, s), lo, hi)
    return dict(num=np.array(num), te_pos=np.array(tp, dtype=int), n_pos=np.array(np_, dtype=int), lo=lo.astype(np.float32), med=med.astype(np.float32), hi=hi.astype(np.float32),
                mu=c.mean(0).astype(np.float32), sd=(c.std(0) + 1e-6).astype(np.float32), blank_col=np.array(drv_ix[0]))


def glm_transform(X, st, chunk=200_000):
    out = np.empty((len(X), len(st["num"]) + 1), dtype=np.float32)
    for i in range(0, len(X), chunk):
        x = X[i:i + chunk]
        b = x[:, st["num"]].copy()
        b[:, st["te_pos"]] = np.log(np.clip(b[:, st["te_pos"]], 1e-4, 1))
        b[:, st["n_pos"]] = np.log1p(b[:, st["n_pos"]])
        b = np.clip(np.where(np.isnan(b), st["med"], b), st["lo"], st["hi"])
        out[i:i + chunk, :-1] = (b - st["mu"]) / st["sd"]
        out[i:i + chunk, -1] = np.isnan(x[:, int(st["blank_col"])]).astype(np.float32)
    return out


def softmax(L):
    L = L - L.max(1, keepdims=True)
    P = np.exp(L)
    return P / P.sum(1, keepdims=True)


def fit_glm(f, Xt, yt, Xm, ym, Xv):
    st = glm_stats(Xt, SEED + f)
    Zt = glm_transform(Xt, st)
    m = LogisticRegression(C=0.3, max_iter=25 if SMOKE else 200, tol=1e-3)
    m.fit(Zt, yt)
    it = int(np.max(m.n_iter_))
    del Zt
    gc.collect()
    np.savez_compressed(MODEL_DIR / f"glm_f{f}.npz", coef=m.coef_.astype(np.float32), intercept=m.intercept_.astype(np.float32), **st)
    def pred(X):
        Z = glm_transform(X, st)
        L = np.empty((len(Z), K), dtype=np.float32)
        for i in range(0, len(Z), 200_000):
            L[i:i + 200_000] = Z[i:i + 200_000] @ m.coef_.T.astype(np.float32) + m.intercept_.astype(np.float32)
        return softmax(L)
    return (pred(Xv), pred(Xm), pred(Xtest)), it


FIT = dict(lgbm=fit_lgbm, xgb=fit_xgb, cat=fit_cat, glm=fit_glm)
for f, (a, b) in enumerate(folds):
    if SMOKE and f > 0:
        break
    todo = [m for m in MODELS if not (PRED_DIR / f"{m}_f{f}.npz").exists()]
    if not todo:
        log(f"fold {f}: semua model sudah selesai, dilewati")
        continue
    if SMOKE:
        rs = np.random.RandomState(0)
        a, b = np.sort(rs.choice(a, 60_000, replace=False)), np.sort(rs.choice(b, 20_000, replace=False))
    prior = np.bincount(y[a], minlength=K) / len(a)
    inner = np.random.RandomState(100 + f).randint(0, TE_INNER, len(a))
    Ta, Tb = [], []
    for key, alpha, _ in ENT:
        cd, _, nc = codes[key]
        oof, C = te_oof(cd[a], y[a], nc, prior, alpha, inner)
        Ta.append(oof)
        Tb.append(te_apply(C, cd[b], prior, alpha))
        del C, oof
    Xt = np.hstack([Xa[a], np.hstack(Ta)[:, te_keep]]).astype(np.float32)
    Xv = np.hstack([Xa[b], np.hstack(Tb)[:, te_keep]]).astype(np.float32)
    del Ta, Tb
    gc.collect()
    blank(Xt, np.where(np.random.RandomState(1000 + f).random(len(a)) < NO_HISTORY)[0], prior)
    bm = np.random.RandomState(2000 + f).random(len(b)) < NO_HISTORY
    Xm = Xv.copy()
    blank(Xm, np.where(bm)[0], prior)
    np.save(RUN / f"val_idx_f{f}.npy", b)
    np.save(RUN / f"val_blank_f{f}.npy", bm)
    yt, ym = y[a], y[b]
    log(f"fold {f}: matriks latih {Xt.shape}, validasi {Xv.shape}, model: {todo}")
    for name in todo:
        t1 = time.time()
        (Pv, Pm, Ps), it = FIT[name](f, Xt, yt, Xm, ym, Xv)
        np.savez_compressed(PRED_DIR / f"{name}_f{f}.npz", v=np.asarray(Pv, dtype=np.float32), m=np.asarray(Pm, dtype=np.float32), s=np.asarray(Ps, dtype=np.float32))
        log(f"fold {f} {name}: iter={it} logloss validasi={log_loss(ym, Pv, labels=LABELS):.5f} | meniru test={log_loss(ym, Pm, labels=LABELS):.5f} | {(time.time() - t1) / 60:.1f} menit")
        del Pv, Pm, Ps
        gc.collect()
    del Xt, Xv, Xm
    gc.collect()
log("training selesai")
