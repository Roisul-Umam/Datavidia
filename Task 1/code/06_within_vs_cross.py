"""Untuk tiap entitas: apakah label train lebih terkait dengan log baris lain di train atau log baris di test?

Ukuran sampel disamakan: proksi dihitung dari m baris acak di tiap sisi, sehingga perbedaan korelasi
tidak berasal dari perbedaan jumlah data.
"""
import sys

import numpy as np
import polars as pl

sys.path.insert(0, r"E:\Kuliyah\Datavidia\Task 1\code")
from features import BASE, CLASSES, K, TARGET, build_all

pl.Config.set_tbl_rows(70); pl.Config.set_tbl_cols(12); pl.Config.set_tbl_width_chars(150); pl.Config.set_float_precision(4)
rng = np.random.default_rng(42)
SIG = ["inv_has_klaim", "inv_dtp_max", "sc_n_upaya_antar", "sc_q_mean", "sc_lgap_mean", "rt_lratio_mean", "sc_man_rate", "jumlah_catatan", "is_cod"]
tr, te, _ = build_all()
keep = ["customer_id", "courier_id", "hub_id", TARGET] + SIG
T = tr.select(keep).with_columns(pl.Series("half", rng.random(tr.height) < 0.5))
X = te.select(["customer_id", "courier_id", "hub_id"] + SIG)


def mean_of_m(src: pl.DataFrame, key: str, m: int) -> pl.DataFrame:
    s = src.select([key] + SIG).with_columns(pl.Series("__r", rng.random(src.height))).sort("__r")
    s = s.group_by(key, maintain_order=True).head(m)
    return s.group_by(key).agg([pl.len().alias("__m")] + [pl.col(c).cast(pl.Float64).mean().alias(c) for c in SIG]).filter(pl.col("__m") == m).drop("__m")


rows = []
for key, m in [("customer_id", 1), ("courier_id", 10), ("hub_id", 200)]:
    A = T.filter(pl.col("half")).select(key, TARGET)
    B = T.filter(~pl.col("half"))
    pin, pcr = mean_of_m(B, key, m), mean_of_m(X, key, m)
    both = A.join(pin, on=key).join(pcr, on=key, suffix="__x")
    for c in range(1, K):
        t = (both[TARGET] == CLASSES[c]).cast(pl.Float64).to_numpy()
        for s in SIG:
            a, b = both[s].fill_null(0).to_numpy(), both[f"{s}__x"].fill_null(0).to_numpy()
            rows.append((key.replace("_id", ""), CLASSES[c][:12], s, float(np.corrcoef(t, a)[0, 1]), float(np.corrcoef(t, b)[0, 1]), both.height))
    print(f"{key}: m={m}, baris uji={both.height}", flush=True)

r = pl.DataFrame(rows, schema=["entitas", "kelas", "sinyal", "r_dalam_split", "r_lintas_split", "n"], orient="row")
r = r.with_columns((pl.col("r_lintas_split") / pl.col("r_dalam_split")).alias("rasio_bertahan"))
for e in ("customer", "courier", "hub"):
    print(f"\n=== {e}: pasangan terkuat ===")
    print(r.filter(pl.col("entitas") == e).sort(pl.col("r_dalam_split").abs(), descending=True).head(14).drop("entitas", "n"))
    k = r.filter((pl.col("entitas") == e) & (pl.col("r_dalam_split").abs() > 0.03))
    print(f"ringkasan {e}: pasangan={k.height}, rata-rata |r| dalam={k['r_dalam_split'].abs().mean():.4f}, lintas={k['r_lintas_split'].abs().mean():.4f}")
r.write_csv(BASE / "exp" / "diagnostics" / "06_within_vs_cross.csv")
