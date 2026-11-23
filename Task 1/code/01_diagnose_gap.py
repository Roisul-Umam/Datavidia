"""Diagnosa selisih validasi lama (0,6187) dengan leaderboard (0,59364).

Hipotesis yang diuji tanpa memakai label test:
  H1  efek entitas (pelanggan, kurir, hub) tidak terbawa ke test
  H2  pergeseran fitur pada log operasional test
  H3  prediksi test berperilaku berbeda dari prediksi OOF (bug inferensi)
"""
import json
from pathlib import Path

import numpy as np
import polars as pl

BASE = Path(r"E:\Kuliyah\Datavidia\Task 1")
D = BASE / "datasets"
OUT = BASE / "exp" / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)
CLASSES = ["normal", "konfirmasi_alamat", "tertunda_operasional", "gagal_antar_awal",
           "retur_pengirim", "rusak_perjalanan", "klaim_hilang"]
K = len(CLASSES)
rng = np.random.default_rng(42)


def load(split):
    s = pl.read_csv(D / split / "shipments.csv", infer_schema_length=100_000)
    sc = (pl.scan_csv(D / split / "scan_events.csv")
          .group_by("shipment_id")
          .agg(pl.len().alias("sc_n"),
               pl.col("gap_hours").mean().alias("gap_mean"),
               pl.col("gap_hours").log1p().mean().alias("lgap_mean"),
               pl.col("gap_hours").max().alias("gap_max"),
               pl.col("scan_quality").mean().alias("q_mean"),
               pl.col("scan_quality").min().alias("q_min"),
               pl.col("is_manual").mean().alias("man_rate"),
               (pl.col("event_type") == "upaya_antar").sum().alias("n_upaya"),
               (pl.col("event_type") == "transit").sum().alias("n_transit"),
               (pl.col("event_type") == "selesai").sum().alias("n_selesai"),
               pl.col("event_type").last().alias("ev_last"))
          .collect(engine="streaming"))
    rt = (pl.scan_csv(D / split / "route_segments.csv")
          .group_by("shipment_id")
          .agg(pl.len().alias("rt_n"), pl.col("durasi_jam").sum().alias("dur_sum"),
               pl.col("durasi_jam").null_count().alias("dur_null"),
               pl.col("biaya_segmen").sum().alias("cost_sum"))
          .collect(engine="streaming"))
    iv = (pl.scan_csv(D / split / "invoices.csv", schema_overrides={"discount": pl.Float64})
          .group_by("shipment_id")
          .agg(pl.len().alias("inv_n"),
               (pl.col("invoice_type") == "klaim").sum().alias("inv_klaim"),
               (pl.col("invoice_type") == "langganan").sum().alias("inv_langganan"),
               pl.col("days_to_pay").max().alias("dtp_max"),
               pl.col("days_to_pay").mean().alias("dtp_mean"),
               pl.col("is_penalty").max().alias("penalty"),
               (pl.col("payment_status") == "gagal").sum().alias("pay_gagal"),
               (pl.col("payment_status") == "tertunda").sum().alias("pay_tunda"),
               pl.col("amount").sum().alias("amt_sum"))
          .collect(engine="streaming"))
    return s.join(sc, on="shipment_id", how="left").join(rt, on="shipment_id", how="left").join(iv, on="shipment_id", how="left")


print("memuat tabel ...", flush=True)
tr, te = load("train"), load("test")
y = tr["status_penanganan"].replace_strict(CLASSES, list(range(K)), return_dtype=pl.Int8).to_numpy().astype(int)
print(tr.shape, te.shape, flush=True)

# -------------------------------------------------------------------------
# H2: pergeseran tiap fitur (AUC satu variabel, 0.5 = tidak bergeser)
# -------------------------------------------------------------------------
def auc_shift(a, b):
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    a = rng.choice(a, min(len(a), 300_000), replace=False)
    b = rng.choice(b, min(len(b), 300_000), replace=False)
    x = np.concatenate([a, b])
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    # rank rata-rata untuk nilai sama
    uniq, inv, cnt = np.unique(sx, return_inverse=True, return_counts=True)
    cum = np.cumsum(cnt)
    avg = cum - (cnt - 1) / 2
    ranks = np.empty(len(x))
    ranks[order] = avg[inv]
    return (ranks[len(a):].sum() - len(b) * (len(b) + 1) / 2) / (len(a) * len(b))


num = [c for c, t in te.schema.items() if t.is_numeric()]
rows = []
for c in num:
    a = tr[c].cast(pl.Float64).to_numpy(); b = te[c].cast(pl.Float64).to_numpy()
    rows.append((c, auc_shift(a, b), np.nanmean(a), np.nanmean(b), np.nanstd(a), np.isnan(a).mean(), np.isnan(b).mean()))
shift = pl.DataFrame(rows, schema=["fitur", "auc", "mean_train", "mean_test", "sd_train", "null_train", "null_test"], orient="row")
shift = shift.with_columns(((pl.col("mean_test") - pl.col("mean_train")) / pl.col("sd_train")).alias("smd"),
                           (pl.col("auc") - 0.5).abs().alias("dev")).sort("dev", descending=True)
with pl.Config(tbl_rows=25, tbl_cols=9, float_precision=4, tbl_width_chars=140):
    print("\n=== H2: fitur paling bergeser (train vs test) ===")
    print(shift.drop("dev").head(25))
shift.write_csv(OUT / "feature_shift.csv")

# -------------------------------------------------------------------------
# H1: apakah efek entitas pada fitur bertahan dari train ke test?
# split-half train vs train->test, dikoreksi dengan ukuran sampel yang sama
# -------------------------------------------------------------------------
def entity_persistence(key, cols, min_n=20):
    out = []
    n = tr.height
    half = rng.random(n) < te.height / n          # pseudo-test berukuran sama dengan test
    a = tr.filter(~pl.Series(half))
    b = tr.filter(pl.Series(half))
    for c in cols:
        def means(df):
            return df.group_by(key).agg(pl.col(c).cast(pl.Float64).mean().alias("m"), pl.len().alias("n")).filter(pl.col("n") >= min_n / 4)
        ja = means(a).join(means(b), on=key, suffix="_b")
        jt = means(tr).join(means(te), on=key, suffix="_b")
        out.append((key, c, np.corrcoef(ja["m"], ja["m_b"])[0, 1], np.corrcoef(jt["m"], jt["m_b"])[0, 1], ja.height, jt.height))
    return out


print("\n=== H1: korelasi rata-rata entitas (pseudo-test dari train vs test sungguhan) ===", flush=True)
res = []
for key in ["courier_id", "hub_id"]:
    res += entity_persistence(key, ["lgap_mean", "q_mean", "man_rate", "n_upaya", "inv_klaim", "dtp_mean", "penalty", "pay_gagal", "declared_value", "distance_km"])
p = pl.DataFrame(res, schema=["entitas", "kolom", "r_pseudo_test", "r_test", "n_a", "n_b"], orient="row")
with pl.Config(tbl_rows=30, float_precision=4):
    print(p)
p.write_csv(OUT / "entity_persistence.csv")

# -------------------------------------------------------------------------
# H1b: kolom proksi label. TE label train vs proksi pada baris lain
# -------------------------------------------------------------------------
print("\n=== Proksi label: rata-rata kolom per kelas pada train ===", flush=True)
proxy_cols = ["n_upaya", "inv_klaim", "penalty", "pay_gagal", "pay_tunda", "dtp_max", "q_mean", "man_rate", "lgap_mean", "dur_null", "n_selesai"]
g = tr.select(proxy_cols + ["status_penanganan"]).group_by("status_penanganan").agg(pl.all().cast(pl.Float64).mean()).sort("status_penanganan")
with pl.Config(tbl_cols=14, float_precision=3, tbl_width_chars=170):
    print(g)

print("\n=== H1b: apakah rate label entitas di train memprediksi proksi pada baris lain? ===", flush=True)
def te_vs_proxy(key, cls, proxy, alpha=5.0):
    prior = float((y == cls).mean())
    mask = rng.random(tr.height) < te.height / tr.height
    d = tr.select(key, proxy).with_columns(pl.Series("t", (y == cls).astype(np.int8)), pl.Series("ps", mask))
    fit = d.filter(~pl.col("ps")).group_by(key).agg(pl.col("t").sum().alias("s"), pl.len().alias("n"))
    fit = fit.with_columns(((pl.col("s") + alpha * prior) / (pl.col("n") + alpha)).alias("te"))
    val = d.filter(pl.col("ps")).join(fit, on=key, how="inner")
    r_pseudo = np.corrcoef(val["te"].to_numpy(), val[proxy].cast(pl.Float64).fill_null(0).to_numpy())[0, 1]
    full = d.group_by(key).agg(pl.col("t").sum().alias("s"), pl.len().alias("n")).with_columns(((pl.col("s") + alpha * prior) / (pl.col("n") + alpha)).alias("te"))
    tst = te.select(key, proxy).join(full, on=key, how="inner")
    r_test = np.corrcoef(tst["te"].to_numpy(), tst[proxy].cast(pl.Float64).fill_null(0).to_numpy())[0, 1]
    return (key, CLASSES[cls], proxy, r_pseudo, r_test)


rows = []
for key in ["customer_id", "courier_id", "hub_id"]:
    for cls in range(1, K):
        for proxy in ["n_upaya", "inv_klaim", "penalty", "pay_gagal", "lgap_mean", "q_mean", "dtp_max"]:
            rows.append(te_vs_proxy(key, cls, proxy))
q = pl.DataFrame(rows, schema=["entitas", "kelas", "proksi", "r_pseudo_test", "r_test"], orient="row")
q = q.with_columns((pl.col("r_pseudo_test").abs() - pl.col("r_test").abs()).alias("selisih_abs"))
strong = q.filter(pl.col("r_pseudo_test").abs() > 0.03).sort(pl.col("r_pseudo_test").abs(), descending=True)
with pl.Config(tbl_rows=30, float_precision=4, tbl_width_chars=120):
    print(strong.head(30))
q.write_csv(OUT / "te_proxy_transfer.csv")
print("rata-rata |r| pseudo-test", round(float(strong["r_pseudo_test"].abs().mean()), 4), "| test", round(float(strong["r_test"].abs().mean()), 4))

# -------------------------------------------------------------------------
# H3: perilaku prediksi lama pada test vs OOF
# -------------------------------------------------------------------------
print("\n=== H3: prediksi lama (OOF vs test) ===", flush=True)
oof = np.load(BASE / "models" / "oof_predictions.npz")
P_oof = oof["proba"].astype(np.float32) / 65535
y_oof = oof["y"].astype(int)
P_te = np.load(BASE / "models" / "reference_test_proba.npy")
ent = lambda P: -(P * np.log(np.clip(P, 1e-9, 1))).sum(1)
print("rata-rata proba OOF ", np.round(P_oof.mean(0), 4))
print("rata-rata proba test", np.round(P_te.mean(0), 4))
print("sebaran kelas train ", np.round(np.bincount(y_oof, minlength=K) / len(y_oof), 4))
print(f"entropi OOF {ent(P_oof).mean():.4f} | test {ent(P_te).mean():.4f}")
print(f"proba maks OOF {P_oof.max(1).mean():.4f} | test {P_te.max(1).mean():.4f}")
for t in (0.5, 0.7, 0.9):
    print(f"porsi proba maks > {t}: OOF {(P_oof.max(1) > t).mean():.4f} | test {(P_te.max(1) > t).mean():.4f}")
