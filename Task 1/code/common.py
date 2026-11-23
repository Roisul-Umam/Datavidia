"""Metrik, pengali kelas, dan target encoding yang dipakai semua eksperimen."""
from __future__ import annotations

import numpy as np

K = 7


def f1_from_pred(y, pred, w=None):
    idx = y.astype(np.int64) * K + pred.astype(np.int64)
    cm = np.bincount(idx, weights=w, minlength=K * K).reshape(K, K).astype(np.float64)
    tp = np.diag(cm)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    f = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(K), where=(2 * tp + fp + fn) > 0)
    return float(f.mean()), f


def apply_weights(P, w):
    return (np.log(np.clip(P, 1e-9, 1)) + np.log(w)).argmax(1)


def optimize_class_weights(P, y, sample_weight=None, passes=3, grid=None, w0=None):
    """Coordinate ascent pada pengali log-probabilitas per kelas untuk Macro F1."""
    grid = np.round(np.arange(0.5, 4.01, 0.05), 3) if grid is None else grid
    lp = np.log(np.clip(P, 1e-9, 1)).astype(np.float32)
    w = np.zeros(K, dtype=np.float32) if w0 is None else np.log(w0).astype(np.float32)
    best, _ = f1_from_pred(y, (lp + w).argmax(1), sample_weight)
    lg = np.log(grid).astype(np.float32)
    for _ in range(passes):
        improved = False
        for c in range(1, K):
            cur, bv, bs = w[c], w[c], best
            others = lp + w
            others[:, c] = -np.inf
            om, oa = others.max(1), others.argmax(1)
            for g in lg:
                s, _ = f1_from_pred(y, np.where(lp[:, c] + g > om, c, oa), sample_weight)
                if s > bs + 1e-6:
                    bs, bv = s, g
            if bv != cur:
                w[c], best, improved = bv, bs, True
        if not improved:
            break
    return np.exp(w).astype(np.float64), best


def crossfit_f1(P, y, sample_weight=None, n_splits=2, seed=7, max_fit=250_000):
    """Macro F1 dengan pengali kelas yang dipelajari pada satu bagian dan dinilai pada bagian lain."""
    rng = np.random.RandomState(seed)
    part = rng.randint(0, n_splits, len(y))
    pred = np.zeros(len(y), dtype=np.int8)
    for h in range(n_splits):
        fit = np.where(part != h)[0]
        if len(fit) > max_fit:
            fit = rng.choice(fit, max_fit, replace=False)
        sw = None if sample_weight is None else sample_weight[fit]
        wgt, _ = optimize_class_weights(P[fit], y[fit], sw, passes=2)
        m = part == h
        pred[m] = apply_weights(P[m], wgt)
    return f1_from_pred(y, pred, sample_weight), pred


def n_bucket(n, cap=6):
    return np.minimum(np.asarray(n, dtype=np.int64), cap)


def test_like_weights(n_val, n_test, cap=6):
    """Bobot baris agar sebaran jumlah saudara di validasi meniru sebaran di test."""
    bv, bt = n_bucket(n_val, cap), n_bucket(n_test, cap)
    pv = np.bincount(bv, minlength=cap + 1) / len(bv)
    pt = np.bincount(bt, minlength=cap + 1) / len(bt)
    ratio = np.divide(pt, pv, out=np.zeros_like(pt), where=pv > 0)
    return ratio[bv]


def te_oof(codes, y, n_cat, prior, alpha, folds):
    C = np.zeros((n_cat, K), dtype=np.int32)
    np.add.at(C, (codes, y), 1)
    out = np.zeros((len(codes), K + 1), dtype=np.float32)
    for f in range(int(folds.max()) + 1):
        m = folds == f
        Cf = np.zeros((n_cat, K), dtype=np.int32)
        np.add.at(Cf, (codes[m], y[m]), 1)
        rest = C[codes[m]] - Cf[codes[m]]
        n = rest.sum(1, keepdims=True)
        out[m, :K] = (rest + alpha * prior) / (n + alpha)
        out[m, K] = n[:, 0]
    return out, C


def te_apply(C, codes, prior, alpha):
    n = C[codes].sum(1, keepdims=True)
    return np.hstack([(C[codes] + alpha * prior) / (n + alpha), n]).astype(np.float32)
