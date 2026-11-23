"""Fitur lanjutan berdasarkan temuan mekanisme data.

hist_*  : riwayat pelanggan di dalam split = rata-rata saudara, atau nilai sendiri bila tidak punya saudara.
cx_*    : rata-rata log pelanggan yang sama di split LAIN (menguji efek pelanggan yang menetap).
load_*  : beban kerja hub/kurir di dalam split, dinormalkan dengan ukuran split.
"""
from __future__ import annotations

import gc
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, r"E:\Kuliyah\Datavidia\Task 1\code")
from features import CACHE, CS_SIGNALS, build_all

CX_SIGNALS = ["inv_has_klaim", "inv_dtp_max", "inv_dtp_mean", "inv_penalty", "sc_n_upaya_antar", "sc_q_mean", "sc_lgap_mean", "sc_man_rate",
              "rt_lratio_mean", "jumlah_catatan", "flag_alamat_baru", "is_cod", "sc_last_selesai", "log_value"]
LOAD_KEYS = {
    "hm": ["hub_id", "ship_month"],
    "cm": ["courier_id", "ship_month"],
    "hmd": ["hub_id", "ship_month", "ship_dow"],
    "hdh": ["hub_id", "ship_dow", "ship_hour"],
    "cd": ["courier_id", "ship_dow"],
    "ch": ["courier_id", "ship_hour"],
    "m": ["ship_month"],
    "md": ["ship_month", "ship_dow"],
}


def build_ext(force=False):
    ptr, pte = CACHE / "fx_train.parquet", CACHE / "fx_test.parquet"
    if not force and ptr.exists() and pte.exists():
        return pl.read_parquet(ptr), pl.read_parquet(pte)
    t0 = time.time()
    tr, te, _ = build_all()
    frames = {"train": tr, "test": te}
    out = {}
    for s, o in (("train", "test"), ("test", "train")):
        df, other = frames[s], frames[o]
        cx = other.group_by("customer_id").agg([pl.len().cast(pl.Float32).alias("cx_n")] +
                                               [pl.col(c).cast(pl.Float64).mean().cast(pl.Float32).alias(f"cx_{c}") for c in CX_SIGNALS])
        d = df.select(["shipment_id", "customer_id", "hub_id", "courier_id", "ship_month", "ship_dow", "ship_hour"] +
                      [c for c in CS_SIGNALS] + [f"cs_loo_{c}" for c in CS_SIGNALS]).join(cx, on="customer_id", how="left")
        hist = [pl.coalesce(pl.col(f"cs_loo_{c}"), pl.col(c).cast(pl.Float32)).alias(f"hist_{c}") for c in CS_SIGNALS]
        d = d.with_columns(hist).with_columns(pl.col("cx_n").fill_null(0))
        scale = 1e6 / df.height
        for name, keys in LOAD_KEYS.items():
            d = d.with_columns((pl.len().over(keys).cast(pl.Float64) * scale).cast(pl.Float32).alias(f"load_{name}"))
        d = d.with_columns(
            (pl.col("load_hm") / pl.col("load_m")).alias("load_hm_share"),
            (pl.col("load_cm") / pl.col("load_m")).alias("load_cm_share"),
        )
        keep = ["shipment_id"] + [c for c in d.columns if c.startswith(("hist_", "cx_", "load_"))]
        out[s] = d.select(keep).sort("shipment_id")
        out[s].write_parquet(ptr if s == "train" else pte, compression="zstd", compression_level=3)
        print(f"  ext {s}: {out[s].shape} ({time.time() - t0:.0f}s)", flush=True)
        del d, cx
        gc.collect()
    return out["train"], out["test"]


if __name__ == "__main__":
    a, b = build_ext(force=True)
    print(a.shape, b.shape)
