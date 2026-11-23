"""Apakah label sebuah baris lebih terkait dengan log baris sendiri atau baris saudara (pelanggan sama)?

own  = korelasi 1[y_i = c] dengan proksi baris i
sib  = korelasi 1[y_i = c] dengan rata-rata proksi baris lain milik pelanggan yang sama, di split yang sama
Bila tabel log tertukar antar saudara, own melemah saat jumlah baris pelanggan naik dan sib mendekati own.
"""
from pathlib import Path

import numpy as np
import polars as pl

BASE = Path(r"E:\Kuliyah\Datavidia\Task 1")
D = BASE / "datasets"
CLASSES = ["normal", "konfirmasi_alamat", "tertunda_operasional", "gagal_antar_awal",
           "retur_pengirim", "rusak_perjalanan", "klaim_hilang"]
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(16); pl.Config.set_tbl_width_chars(170); pl.Config.set_float_precision(4)


def load(split):
    cols = ["shipment_id", "customer_id", "courier_id", "hub_id", "ship_month"] + (["status_penanganan"] if split == "train" else [])
    s = pl.read_csv(D / split / "shipments.csv", columns=cols)
    sc = (pl.scan_csv(D / split / "scan_events.csv").group_by("shipment_id").agg(
        pl.col("scan_quality").mean().alias("q_mean"), pl.col("gap_hours").log1p().mean().alias("lgap"),
        (pl.col("event_type") == "upaya_antar").sum().cast(pl.Float64).alias("n_upaya"),
        pl.col("is_manual").mean().alias("man_rate")).collect(engine="streaming"))
    iv = (pl.scan_csv(D / split / "invoices.csv", schema_overrides={"discount": pl.Float64}).group_by("shipment_id").agg(
        (pl.col("invoice_type") == "klaim").sum().cast(pl.Float64).alias("inv_klaim"),
        pl.col("days_to_pay").max().cast(pl.Float64).alias("dtp_max"),
        pl.col("is_penalty").max().cast(pl.Float64).alias("penalty")).collect(engine="streaming"))
    rt = (pl.scan_csv(D / split / "route_segments.csv").group_by("shipment_id").agg(
        pl.col("durasi_jam").sum().alias("dur"), pl.len().cast(pl.Float64).alias("rt_n")).collect(engine="streaming"))
    return s.join(sc, on="shipment_id", how="left").join(iv, on="shipment_id", how="left").join(rt, on="shipment_id", how="left")


tr = load("train")
te = load("test")
PROX = ["q_mean", "lgap", "n_upaya", "man_rate", "inv_klaim", "dtp_max", "penalty", "dur", "rt_n"]
pairs = [("gagal_antar_awal", "q_mean"), ("rusak_perjalanan", "q_mean"), ("tertunda_operasional", "lgap"), ("konfirmasi_alamat", "n_upaya"),
         ("retur_pengirim", "inv_klaim"), ("klaim_hilang", "inv_klaim"), ("tertunda_operasional", "dtp_max"), ("tertunda_operasional", "dur"),
         ("rusak_perjalanan", "rt_n"), ("gagal_antar_awal", "man_rate")]

for keyname in ("customer_id", "courier_id"):
    g = tr.group_by(keyname).agg([pl.len().alias("n")] + [pl.col(p).sum().alias(f"{p}__s") for p in PROX])
    d = tr.join(g, on=keyname)
    print(f"\n########## Kunci: {keyname} (saudara = baris train lain dengan {keyname} sama) ##########", flush=True)
    rows = []
    buckets = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 99)] if keyname == "customer_id" else [(1, 20), (21, 40), (41, 60), (61, 9999)]
    for lo, hi in buckets:
        m = d.filter(pl.col("n").is_between(lo, hi))
        for cname, p in pairs:
            t = (m["status_penanganan"] == cname).cast(pl.Int8).to_numpy().astype(float)
            own = m[p].fill_null(0).to_numpy()
            r_own = np.corrcoef(t, own)[0, 1]
            if hi > 1:
                k = m.filter(pl.col("n") > 1)
                tt = (k["status_penanganan"] == cname).cast(pl.Int8).to_numpy().astype(float)
                sib = ((k[f"{p}__s"] - k[p].fill_null(0)) / (k["n"] - 1)).to_numpy()
                r_sib = np.corrcoef(tt, sib)[0, 1]
            else:
                r_sib = np.nan
            rows.append((f"{lo}-{hi}", cname[:10], p, r_own, r_sib))
    out = pl.DataFrame(rows, schema=["n_baris", "kelas", "proksi", "own", "sib"], orient="row")
    w = out.with_columns(pl.col("own").round(3), pl.col("sib").round(3))
    print(w.pivot(on="n_baris", index=["kelas", "proksi"], values="own").rename(lambda c: f"own_{c}" if c[0].isdigit() else c))
    print(w.pivot(on="n_baris", index=["kelas", "proksi"], values="sib").rename(lambda c: f"sib_{c}" if c[0].isdigit() else c))
