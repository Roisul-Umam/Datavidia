"""Fitur penggerak riwayat pelanggan, mengikuti hasil pemetaan tipe sinyal.

Tipe berasal dari uji pasangan pada train (13_signal_typing.py dan 13b_row_attr_typing.py):
  rata2    label mengikuti rata-rata seluruh baris train pelanggan
  saudara  label mengikuti rata-rata baris train lain milik pelanggan
  campuran efek baris sendiri ditambah efek baris saudara

Baris train : rata2 = rata-rata termasuk dirinya, saudara/campuran = rata-rata tanpa dirinya.
Baris test  : seluruh tipe = rata-rata baris TRAIN pelanggan. Log baris test tidak ikut dihitung.
Pelanggan tanpa baris train bernilai kosong.
"""
from __future__ import annotations

import sys
import time

import polars as pl

sys.path.insert(0, r"E:\Kuliyah\Datavidia\Task 1\code")
from features import BASE, CACHE, build_all

DIAG = BASE / "exp" / "diagnostics"
MIN_SIBLING_EFFECT = 0.02


def load_types() -> pl.DataFrame:
    return pl.concat([pl.read_csv(DIAG / "13_signal_typing.csv"), pl.read_csv(DIAG / "13b_row_attr_typing.csv")]).unique("sinyal", keep="first", maintain_order=True)


def signal_plan(columns) -> tuple[dict, list]:
    t = load_types()
    plan, own_drop = {}, []
    for r in t.iter_rows(named=True):
        c, k = r["sinyal"], r["tipe"]
        if c not in columns:
            continue
        if k == "rata2":
            plan[c] = "mean"
            own_drop.append(c)
        elif k == "saudara":
            plan[c] = "loo"
            own_drop.append(c)
        elif k == "campuran" or (k == "sendiri" and r["kuat_saudara"] >= MIN_SIBLING_EFFECT):
            plan[c] = "loo"
    # Seluruh isi tagihan bekerja di level pelanggan, jadi nilai baris sendiri tidak dipakai.
    own_drop += [c for c in columns if (c.startswith("inv_") or c.startswith("klaim_")) and c not in own_drop]
    own_drop += [c for c in ("attempts_vs_customer", "q_vs_customer") if c in columns]
    return plan, sorted(set(own_drop))


def build_driver(force=False):
    ptr, pte = CACHE / "fd_train.parquet", CACHE / "fd_test.parquet"
    tr, te, _ = build_all()
    plan, own_drop = signal_plan(set(te.columns))
    if not force and ptr.exists() and pte.exists():
        return pl.read_parquet(ptr), pl.read_parquet(pte), {"plan": plan, "own_drop": own_drop}
    t0 = time.time()
    sig = list(plan)
    g = tr.group_by("customer_id").agg(
        pl.len().cast(pl.Float64).alias("__n"),
        *[pl.col(c).cast(pl.Float64).sum().alias(f"__s_{c}") for c in sig],
        *[pl.col(c).is_not_null().sum().cast(pl.Float64).alias(f"__c_{c}") for c in sig])
    a = tr.select(["shipment_id", "customer_id"] + sig).join(g, on="customer_id", how="left", maintain_order="left")
    ex = [(pl.col("__n") - 1).cast(pl.Float32).alias("drv_n")]
    for c, kind in plan.items():
        s, n, v = pl.col(f"__s_{c}"), pl.col(f"__c_{c}"), pl.col(c).cast(pl.Float64)
        if kind == "mean":
            e = pl.when(n > 0).then(s / n).otherwise(None)
        else:
            rest = n - v.is_not_null().cast(pl.Float64)
            e = pl.when(rest > 0).then((s - v.fill_null(0)) / rest).otherwise(None)
        ex.append(e.cast(pl.Float32).alias(f"drv_{c}"))
    d_tr = a.select(["shipment_id"] + ex)
    b = te.select("shipment_id", "customer_id").join(g, on="customer_id", how="left", maintain_order="left")
    ex = [pl.col("__n").fill_null(0).cast(pl.Float32).alias("drv_n")]
    for c in plan:
        s, n = pl.col(f"__s_{c}"), pl.col(f"__c_{c}")
        ex.append(pl.when(n > 0).then(s / n).otherwise(None).cast(pl.Float32).alias(f"drv_{c}"))
    d_te = b.select(["shipment_id"] + ex)
    assert (d_tr["shipment_id"] == tr["shipment_id"]).all() and (d_te["shipment_id"] == te["shipment_id"]).all()
    d_tr.write_parquet(ptr, compression="zstd", compression_level=3)
    d_te.write_parquet(pte, compression="zstd", compression_level=3)
    print(f"fitur penggerak: {len(plan)} sinyal, buang {len(own_drop)} kolom baris sendiri ({time.time() - t0:.0f}s)", flush=True)
    return d_tr, d_te, {"plan": plan, "own_drop": own_drop}


if __name__ == "__main__":
    x, y, info = build_driver(force=True)
    print(x.shape, y.shape)
    print("mean:", [k for k, v in info["plan"].items() if v == "mean"])
    print("loo :", [k for k, v in info["plan"].items() if v == "loo"])
    print("buang:", info["own_drop"])
