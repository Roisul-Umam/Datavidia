"""Sidik jari log yang lebih rinci: per jenis event, per perangkat, per posisi, bigram urutan, per segmen rute, per jenis tagihan."""
from __future__ import annotations

import gc
import sys
import time

import polars as pl

sys.path.insert(0, r"E:\Kuliyah\Datavidia\Task 1\code")
from features import CACHE, DATA, DEVICES, EVENTS, INV_TYPES, MODAS, nominal_speeds, safe_div, scan

PAY_METHODS = ["transfer", "ewallet", "kartu", "cod", "tunai"]


def sig_scan(split: str) -> pl.DataFrame:
    df = (scan(split, "scan_events").select("shipment_id", "event_type", "gap_hours", "device_type", "scan_quality", "is_manual")
          .with_columns(pl.col("gap_hours").log1p().alias("lg")).collect())
    e, g, lg, q, m, d = [pl.col(c) for c in ("event_type", "gap_hours", "lg", "scan_quality", "is_manual", "device_type")]
    aggs = []
    for ev in EVENTS:
        k = e == ev
        aggs += [lg.filter(k).mean().alias(f"sg_lgap_{ev}"), g.filter(k).max().alias(f"sg_gapmax_{ev}"), q.filter(k).mean().alias(f"sg_q_{ev}"),
                 q.filter(k).min().alias(f"sg_qmin_{ev}"), m.filter(k).mean().alias(f"sg_man_{ev}")]
    for dv in DEVICES:
        k = d == dv
        aggs += [q.filter(k).mean().alias(f"sg_q_dev_{dv}"), lg.filter(k).mean().alias(f"sg_lgap_dev_{dv}"), m.filter(k).mean().alias(f"sg_man_dev_{dv}")]
    aggs += [(q == v).sum().alias(f"sg_qn_{v}") for v in range(1, 6)]
    aggs += [q.filter(m == 1).mean().alias("sg_q_manual"), q.filter(m == 0).mean().alias("sg_q_auto"),
             lg.filter(m == 1).mean().alias("sg_lgap_manual"), lg.filter(m == 0).mean().alias("sg_lgap_auto")]
    for i in range(3):
        aggs += [q.slice(i, 1).first().alias(f"sg_q_p{i}"), lg.slice(i, 1).first().alias(f"sg_lg_p{i}"), m.slice(i, 1).first().alias(f"sg_m_p{i}")]
    for i in (1, 2):
        aggs += [q.slice(-i, 1).first().alias(f"sg_q_l{i}"), lg.slice(-i, 1).first().alias(f"sg_lg_l{i}"), m.slice(-i, 1).first().alias(f"sg_m_l{i}")]
    aggs += [q.diff().abs().mean().alias("sg_q_tv"), lg.diff().abs().mean().alias("sg_lg_tv"), m.diff().abs().sum().alias("sg_m_switch"),
             d.n_unique().alias("sg_n_dev"), e.n_unique().alias("sg_n_ev")]
    out = df.group_by("shipment_id").agg(aggs)
    bg = (df.select("shipment_id", "event_type").with_columns(pl.col("event_type").shift(1).over("shipment_id").alias("prev")).drop_nulls("prev")
          .with_columns((pl.col("prev") + "_" + pl.col("event_type")).alias("bg")).group_by("shipment_id", "bg").len()
          .pivot(on="bg", index="shipment_id", values="len").fill_null(0))
    bg = bg.rename({c: f"sg_bg_{c}" for c in bg.columns if c != "shipment_id"})
    out = out.join(bg, on="shipment_id", how="left")
    return out.with_columns([pl.col(c).fill_null(0) for c in out.columns if c.startswith("sg_bg_")])


def sig_route(split: str, vmoda: dict, cost_km: dict) -> pl.DataFrame:
    df = (scan(split, "route_segments").select("shipment_id", "urutan", "moda", "jarak_segmen_km", "durasi_jam", "biaya_segmen").collect()
          .sort("shipment_id", "urutan"))
    df = df.with_columns(pl.col("moda").replace_strict(vmoda, return_dtype=pl.Float64).alias("v"),
                         pl.col("moda").replace_strict(cost_km, return_dtype=pl.Float64).alias("ck"))
    df = df.with_columns(
        pl.when(pl.col("durasi_jam") > 0).then((pl.col("durasi_jam") * pl.col("v") / pl.col("jarak_segmen_km")).log()).otherwise(None).alias("lr"),
        (pl.col("biaya_segmen") / (pl.col("jarak_segmen_km") * pl.col("ck"))).log().alias("lc"))
    lr, lc, km, dur = pl.col("lr"), pl.col("lc"), pl.col("jarak_segmen_km"), pl.col("durasi_jam")
    aggs = [lc.mean().alias("sr_lc_mean"), lc.max().alias("sr_lc_max"), lc.min().alias("sr_lc_min"), lc.std().alias("sr_lc_std"),
            (lr - lr.mean()).abs().max().alias("sr_lr_range"), (km / km.sum()).max().alias("sr_km_share_max")]
    for i in range(3):
        aggs += [lr.slice(i, 1).first().alias(f"sr_lr_p{i}"), lc.slice(i, 1).first().alias(f"sr_lc_p{i}"), km.slice(i, 1).first().alias(f"sr_km_p{i}"),
                 dur.slice(i, 1).first().alias(f"sr_dur_p{i}")]
    for x in MODAS:
        k = pl.col("moda") == x
        aggs += [lr.filter(k).mean().alias(f"sr_lr_{x}"), dur.filter(k).sum().alias(f"sr_dur_{x}")]
    return df.group_by("shipment_id").agg(aggs)


def sig_invoice(split: str, ship: pl.DataFrame, cust: pl.DataFrame) -> pl.DataFrame:
    df = (scan(split, "invoices").collect().join(ship, on="shipment_id", how="left").join(cust, on="customer_id", how="left"))
    t, a, ds, tx, dtp, st, pen, im, pm = [pl.col(c) for c in ("invoice_type", "amount", "discount", "tax", "days_to_pay", "payment_status", "is_penalty", "issued_month", "payment_method")]
    df = df.with_columns((a / pl.col("declared_value")).alias("av"), (tx / a).alias("tr"), (ds / a).alias("dr"), ((im - pl.col("ship_month")) % 12).alias("lag"),
                         (pm == pl.col("preferred_payment")).cast(pl.Int8).alias("pref"))
    av, tr, dr, lag, pref = [pl.col(c) for c in ("av", "tr", "dr", "lag", "pref")]
    aggs = [tr.mean().alias("si_tax_rate"), dr.mean().alias("si_disc_rate"), av.mean().alias("si_amt_value"), pref.mean().alias("si_pref"),
            ds.null_count().alias("si_disc_null")]
    aggs += [(pm == x).sum().alias(f"si_pm_{x}") for x in PAY_METHODS]
    aggs += [(st == x).sum().alias(f"si_st_{x}") for x in ("lunas", "tertunda", "gagal")]
    for x in INV_TYPES:
        k = t == x
        aggs += [av.filter(k).mean().alias(f"si_av_{x}"), tr.filter(k).mean().alias(f"si_tr_{x}"), pen.filter(k).max().alias(f"si_pen_{x}"),
                 (st == "gagal").filter(k).sum().alias(f"si_gagal_{x}"), (st == "tertunda").filter(k).sum().alias(f"si_tunda_{x}")]
    aggs += [lag.filter(t == "klaim").max().alias("si_klaim_lag")]
    return df.group_by("shipment_id").agg(aggs)


def build_sig(force=False):
    ptr, pte = CACHE / "fs_train.parquet", CACHE / "fs_test.parquet"
    if not force and ptr.exists() and pte.exists():
        return pl.read_parquet(ptr), pl.read_parquet(pte)
    t0 = time.time()
    vmoda = nominal_speeds()
    r = pl.concat([scan(s, "route_segments").select("moda", "jarak_segmen_km", "biaya_segmen") for s in ("train", "test")])
    c = r.with_columns((pl.col("biaya_segmen") / pl.col("jarak_segmen_km")).alias("x")).group_by("moda").agg(pl.col("x").median()).collect(engine="streaming")
    cost_km = dict(zip(c["moda"].to_list(), c["x"].to_list()))
    cust = pl.read_csv(DATA / "train" / "customers.csv", columns=["customer_id", "preferred_payment"])
    out = {}
    for split in ("train", "test"):
        ship = pl.read_csv(DATA / split / "shipments.csv", columns=["shipment_id", "declared_value", "ship_month", "weight_kg", "volume_l", "berat_volumetrik"])
        ss = sig_scan(split)
        sr = sig_route(split, vmoda, cost_km)
        si = sig_invoice(split, ship.select("shipment_id", "declared_value", "ship_month"), cust)
        d = ship.select("shipment_id", safe_div(pl.col("berat_volumetrik"), pl.col("weight_kg")).alias("ss_bv_w"), safe_div(pl.col("berat_volumetrik"), pl.col("volume_l")).alias("ss_bv_v"))
        d = d.join(ss, on="shipment_id", how="left").join(sr, on="shipment_id", how="left").join(si, on="shipment_id", how="left").sort("shipment_id")
        num = [x for x, tp in d.schema.items() if tp.is_numeric()]
        d = d.with_columns([pl.col(x).cast(pl.Float32) for x in num])
        out[split] = d
        d.write_parquet(ptr if split == "train" else pte, compression="zstd", compression_level=3)
        print(f"  sig {split}: {d.shape} ({time.time() - t0:.0f}s)", flush=True)
        del ss, sr, si, ship
        gc.collect()
    return out["train"], out["test"]


if __name__ == "__main__":
    a, b = build_sig(force=True)
    print(a.shape, b.shape)
    print(sorted(set(a.columns) ^ set(b.columns)))
