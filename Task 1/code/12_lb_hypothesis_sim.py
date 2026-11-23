"""Hipotesis mana yang mereproduksi dua skor leaderboard yang sudah diketahui?

Skor nyata: submission lama 0,59364 dan kandidat sementara 0,48574.

Cara kerja: model proksi P(y | fitur) dilatih pada train. Untuk tiap hipotesis tentang riwayat pelanggan
yang menggerakkan label test, fitur riwayat baris test disusun sesuai hipotesis itu, lalu proksi
memberi peluang kelas. Dari peluang itu dihitung Macro F1 yang DIHARAPKAN untuk kedua submission
yang sudah dinilai. Tidak ada submission baru yang dibutuhkan.
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
from common import f1_from_pred, te_apply, te_oof
from features import BASE, CAT_COLS, CLASSES, CS_MAX, CS_MIN, CS_SIGNALS, DATA, ID_COLS, K, NOISE_COLS, TARGET, build_all, customer_folds
from features_ext import build_ext

LB_OLD, LB_INTERIM = 0.59364, 0.48574
OUT = BASE / "exp" / "diagnostics"
MODELS = BASE / "exp" / "models"
MODELS.mkdir(parents=True, exist_ok=True)
t0 = time.time()


def expected_f1(P, pred):
    tp = np.array([P[pred == k, k].sum() for k in range(K)])
    f = 2 * tp / (np.bincount(pred, minlength=K) + P.sum(0))
    return float(f.mean()), f


tr, te, _ = build_all()
fx_tr, fx_te = build_ext()
y = tr[TARGET].replace_strict(CLASSES, list(range(K)), return_dtype=pl.Int8).to_numpy().astype(np.int64)
fold = customer_folds(tr["customer_id"].rank("dense").to_numpy())
drop = set(ID_COLS + NOISE_COLS + [TARGET, "hub_month", "courier_month"])
base = [c for c in te.columns if c not in drop]
hist = [c for c in fx_tr.columns if c.startswith("hist_")]
cats = [c for c in CAT_COLS if c in base]
va, trn = np.where(fold == 0)[0], np.where(fold != 0)[0]
rng = np.random.RandomState(42)
prior = np.bincount(y[trn], minlength=K) / len(trn)
inner = rng.randint(0, 5, len(trn))
Tt, Tv, Ts, te_names = [], [], [], []
for key, alpha, p in [("courier_id", 10.0, "cr"), ("hub_id", 30.0, "hb"), ("hub_month", 15.0, "hm")]:
    both = pl.concat([tr.select(key), te.select(key)])[key].rank("dense").cast(pl.Int32) - 1
    codes, codes_te = both[:tr.height].to_numpy(), both[tr.height:].to_numpy()
    oof, C = te_oof(codes[trn], y[trn], int(both.max()) + 1, prior, alpha, inner)
    Tt.append(oof)
    Tv.append(te_apply(C, codes[va], prior, alpha))
    Ts.append(te_apply(C, codes_te, prior, alpha))
    te_names += [f"te_{p}_{c[:6]}" for c in CLASSES] + [f"te_{p}_n"]
Tt, Tv, Ts = np.hstack(Tt), np.hstack(Tv), np.hstack(Ts)
names = base + hist + te_names
Xa = np.hstack([tr.select(pl.col(base).cast(pl.Float32)).to_numpy(), fx_tr.select(pl.col(hist).cast(pl.Float32)).to_numpy()])
Xs = np.hstack([te.select(pl.col(base).cast(pl.Float32)).to_numpy(), fx_te.select(pl.col(hist).cast(pl.Float32)).to_numpy(), Ts])
Xt, Xv = np.hstack([Xa[trn], Tt]), np.hstack([Xa[va], Tv])
del Xa, Tt, Tv, Ts
gc.collect()
ix = {n: i for i, n in enumerate(names)}
print(f"matriks siap {Xt.shape} ({time.time() - t0:.0f}s)", flush=True)

params = dict(objective="multiclass", num_class=K, metric="multi_logloss", learning_rate=0.08, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, max_bin=127, num_threads=14, verbose=-1, seed=42)


def fit(keep, label):
    idx = [ix[n] for n in keep]
    dt = lgb.Dataset(Xt[:, idx], y[trn], feature_name=keep, categorical_feature=[c for c in cats if c in keep], free_raw_data=True)
    dv = lgb.Dataset(Xv[:, idx], y[va], reference=dt, free_raw_data=True)
    m = lgb.train(params, dt, 1200, valid_sets=[dv], callbacks=[lgb.early_stopping(40, verbose=False)])
    pv = m.predict(Xv[:, idx], num_iteration=m.best_iteration)
    print(f"proksi {label}: iter={m.best_iteration} logloss={log_loss(y[va], pv, labels=list(range(K))):.5f} ({time.time() - t0:.0f}s)", flush=True)
    m.save_model(str(MODELS / f"proxy_{label}.txt"), num_iteration=m.best_iteration)
    return m, idx, pv


row_only = [n for n in names if not n.startswith(("cs_", "hist_")) and n not in ("attempts_vs_customer", "q_vs_customer")]
m_full, idx_full, pv_full = fit(names, "full")
m_row, idx_row, pv_row = fit(row_only, "row")

# ---- pemeriksaan kalibrasi metode: F1 harapan vs F1 nyata pada fold validasi ----
oof = np.load(BASE / "models" / "oof_predictions.npz")
assert (oof["y"].astype(np.int64) == y).all(), "urutan OOF lama tidak sejajar"
pred_old_oof = oof["pred_crossfit"].astype(np.int64)
print("\n=== Kalibrasi metode pada fold validasi (prediksi model lama sebagai pihak luar) ===")
print(f"F1 nyata {f1_from_pred(y[va], pred_old_oof[va])[0]:.4f} | F1 harapan dari proksi penuh {expected_f1(pv_full, pred_old_oof[va])[0]:.4f}", flush=True)
print(f"F1 nyata argmax proksi {f1_from_pred(y[va], pv_full.argmax(1))[0]:.4f} | harapan {expected_f1(pv_full, pv_full.argmax(1))[0]:.4f}", flush=True)

# ---- susun fitur riwayat test untuk tiap hipotesis ----
sig = [c for c in CS_SIGNALS if c in te.columns]
def agg(df, n, s, mx, mn):
    return df.group_by("customer_id").agg(pl.len().cast(pl.Float64).alias(n), *[pl.col(c).cast(pl.Float64).fill_null(0).sum().alias(f"{s}{c}") for c in sig],
                                          *[pl.col(c).cast(pl.Float64).max().alias(f"{mx}{c}") for c in CS_MAX], *[pl.col(c).cast(pl.Float64).min().alias(f"{mn}{c}") for c in CS_MIN])

J = (te.select(["customer_id"] + sorted(set(sig) | set(CS_MAX) | set(CS_MIN)))
     .join(agg(tr, "m", "S_", "AX_", "AN_"), on="customer_id", how="left", maintain_order="left")
     .join(agg(te, "t", "T_", "BX_", "BN_"), on="customer_id", how="left", maintain_order="left"))
mcnt = J["m"].fill_null(0).to_numpy()
tcnt = J["t"].to_numpy()
has = mcnt > 0
print(f"\nporsi baris test yang punya riwayat di train: {has.mean():.4f}", flush=True)


def fnum(col):
    return J[col].cast(pl.Float64).to_numpy()


def variant(mode):
    out = {}
    for c in sig:
        own = fnum(c)
        own0 = np.nan_to_num(own)
        S = np.nan_to_num(fnum(f"S_{c}"))
        T = fnum(f"T_{c}")
        with np.errstate(divide="ignore", invalid="ignore"):
            if mode == "train+sendiri":
                n, sm = mcnt + 1, S + own0
                loo = np.where(has, S / mcnt, np.nan)
            elif mode in ("train_saja_n=m", "train_saja_n=m+1"):
                n = np.where(has, mcnt if mode.endswith("=m") else mcnt + 1, 1.0)
                sm = np.where(has, S, own0)
                loo = np.where(has, S / mcnt, np.nan)
            elif mode == "train+test":
                n, sm = mcnt + tcnt, S + T
                loo = np.where(n > 1, (sm - own0) / (n - 1), np.nan)
            mean = np.where(has, S / mcnt, own0) if mode.startswith("train_saja") else sm / n
        out[f"cs_mean_{c}"] = mean
        out[f"cs_loo_{c}"] = loo
        out[f"hist_{c}"] = np.where(np.isnan(loo), own, loo)
        if f"cs_sum_{c}" in ix:
            out[f"cs_sum_{c}"] = sm
        out["cs_n"] = n
    for c in CS_MAX:
        a, b, own = fnum(f"AX_{c}"), fnum(f"BX_{c}"), fnum(c)
        out[f"cs_max_{c}"] = np.fmax(a, own) if mode == "train+sendiri" else (np.where(has, a, own) if mode.startswith("train_saja") else np.fmax(a, b))
    for c in CS_MIN:
        a, b, own = fnum(f"AN_{c}"), fnum(f"BN_{c}"), fnum(c)
        out[f"cs_min_{c}"] = np.fmin(a, own) if mode == "train+sendiri" else (np.where(has, a, own) if mode.startswith("train_saja") else np.fmin(a, b))
    out["attempts_vs_customer"] = fnum("sc_n_upaya_antar") - out["cs_mean_sc_n_upaya_antar"]
    out["q_vs_customer"] = fnum("sc_q_mean") - out["cs_mean_sc_q_mean"]
    return out


def to_idx(path):
    d = pl.read_csv(path)
    d = te.select("shipment_id").join(d, on="shipment_id", how="left", maintain_order="left")
    return d["status_penanganan"].replace_strict(CLASSES, list(range(K)), return_dtype=pl.Int8).to_numpy().astype(np.int64)


pred_old = to_idx(BASE / "submission.csv")
pred_int = to_idx(BASE / "exp" / "submissions" / "sub_interim_lgbm.csv")
print(f"kecocokan kedua submission: {(pred_old == pred_int).mean():.4f}", flush=True)

rows, probs = [], {}
def score(label, P):
    fo, per_o = expected_f1(P, pred_old)
    fi, per_i = expected_f1(P, pred_int)
    err = abs(fo - LB_OLD) + abs(fi - LB_INTERIM)
    rows.append(dict(hipotesis=label, f1_lama=round(fo, 4), f1_sementara=round(fi, 4), galat_total=round(err, 4), rerata_proba=np.round(P.mean(0), 3).tolist()))
    print(f"{label:22s} lama={fo:.4f} (nyata {LB_OLD})  sementara={fi:.4f} (nyata {LB_INTERIM})  galat={err:.4f}", flush=True)
    probs[label] = P.astype(np.float32)


print("\n=== Skor harapan kedua submission menurut tiap hipotesis ===")
score("dalam_test", m_full.predict(Xs[:, idx_full]))
score("tanpa_riwayat", m_row.predict(Xs[:, idx_row]))
for mode in ("train+sendiri", "train_saja_n=m", "train_saja_n=m+1", "train+test"):
    X = Xs.copy()
    for k, v in variant(mode).items():
        if k in ix:
            X[:, ix[k]] = v.astype(np.float32)
    score(mode, m_full.predict(X[:, idx_full]))
    del X
    gc.collect()

rows = sorted(rows, key=lambda r: r["galat_total"])
(OUT / "12_lb_hypothesis_sim.json").write_text(json.dumps(rows, indent=2))
np.savez_compressed(BASE / "exp" / "preds" / "hypothesis_test_proba.npz", **{k.replace("+", "_").replace("=", ""): (v * 65535).round().astype(np.uint16) for k, v in probs.items()})
print("\nperingkat hipotesis (galat terkecil lebih baik):")
for r in rows:
    print(r)
print(f"selesai ({(time.time() - t0) / 60:.1f} menit)", flush=True)
