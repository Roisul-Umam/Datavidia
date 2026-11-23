"""Harness eksperimen cepat: fold berkelompok yang sama untuk semua perbandingan fitur."""
from __future__ import annotations

import gc
import json
import sys
import time
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import log_loss

sys.path.insert(0, r"E:\Kuliyah\Datavidia\Task 1\code")
from common import crossfit_f1, f1_from_pred, te_apply, te_oof, test_like_weights
from features import BASE, CAT_COLS, CLASSES, ID_COLS, K, NOISE_COLS, TARGET, build_all, customer_folds
from features_ext import build_ext

PARAMS = dict(objective="multiclass", num_class=K, metric="multi_logloss", learning_rate=0.08, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, max_bin=127, num_threads=14, verbose=-1, seed=42)


def prepare(fold_id=0, n_train=600_000, seed=42):
    tr, te, _ = build_all()
    fx_tr, fx_te = build_ext()
    assert (tr["shipment_id"] == fx_tr["shipment_id"]).all()
    y = tr[TARGET].replace_strict(CLASSES, list(range(K)), return_dtype=pl.Int8).to_numpy().astype(np.int64)
    fold = customer_folds(tr["customer_id"].rank("dense").to_numpy())
    n_all, n_test = tr["cs_n"].to_numpy(), te["cs_n"].to_numpy()
    drop = set(ID_COLS + NOISE_COLS + [TARGET, "hub_month", "courier_month"])
    base = [c for c in te.columns if c not in drop]
    ext = [c for c in fx_tr.columns if c.startswith(("hist_", "cx_"))]
    va = np.where(fold == fold_id)[0]
    trn = np.where(fold != fold_id)[0]
    rng = np.random.RandomState(seed)
    sub = rng.choice(trn, min(n_train, len(trn)), replace=False)
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
    H = SimpleNamespace(tr=tr, te=te, fx_tr=fx_tr, fx_te=fx_te, y=y, fold=fold, n_all=n_all, n_test=n_test, base=base, ext=ext, va=va, trn=trn, sub=sub,
                        yt=y[sub], yv=y[va], Tt=np.hstack(Tt), Tv=np.hstack(Tv), te_names=te_names, cats=[c for c in CAT_COLS if c in base],
                        wv=test_like_weights(n_all[va], n_test))
    H.Xb = tr.select(pl.col(base).cast(pl.Float32)).to_numpy()
    H.Xe = fx_tr.select(pl.col(ext).cast(pl.Float32)).to_numpy()
    return H


def evaluate(H, name, extra=None, extra_names=None, params=None, rounds=1500, sample_weight=None, top=14):
    """extra berukuran n_baris_train, akan diiris sesuai fold."""
    t0 = time.time()
    pt, pv = [H.Xb[H.sub], H.Xe[H.sub], H.Tt], [H.Xb[H.va], H.Xe[H.va], H.Tv]
    names = H.base + H.ext + H.te_names
    if extra is not None:
        pt.append(extra[H.sub])
        pv.append(extra[H.va])
        names = names + list(extra_names)
    Xt, Xv = np.hstack(pt), np.hstack(pv)
    del pt, pv
    dt = lgb.Dataset(Xt, H.yt, weight=sample_weight, feature_name=names, categorical_feature=H.cats, free_raw_data=True)
    dv = lgb.Dataset(Xv, H.yv, reference=dt, free_raw_data=True)
    m = lgb.train(params or PARAMS, dt, rounds, valid_sets=[dv], callbacks=[lgb.early_stopping(40, verbose=False)])
    P = m.predict(Xv, num_iteration=m.best_iteration)
    (f_cf, per), _ = crossfit_f1(P, H.yv)
    (f_w, per_w), _ = crossfit_f1(P, H.yv, H.wv)
    r = dict(iters=m.best_iteration, logloss=round(log_loss(H.yv, P, labels=list(range(K))), 5), f1_argmax=round(f1_from_pred(H.yv, P.argmax(1))[0], 5),
             f1_crossfit=round(f_cf, 5), f1_mirip_test=round(f_w, 5), per_kelas=dict(zip([c[:6] for c in CLASSES], np.round(per_w, 3).tolist())), menit=round((time.time() - t0) / 60, 1))
    print(name, json.dumps(r), flush=True)
    gain = m.feature_importance("gain")
    order = np.argsort(gain)[::-1][:top]
    print("   teratas:", [(names[i], round(100 * gain[i] / gain.sum(), 1)) for i in order], flush=True)
    imp = dict(zip(names, gain.tolist()))
    del dt, dv, m, Xt, Xv
    gc.collect()
    return r, P, imp
