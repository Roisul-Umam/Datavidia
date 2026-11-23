"""Feature store Task 1.

Perbedaan utama dari pipeline lama:
  * Agregat pelanggan dihitung DI DALAM tiap split (train sendiri, test sendiri), karena label
    bergantung pada agregat tagihan dan scan pelanggan di split yang sama.
  * Profil kurir dan hub tetap memakai train + test tanpa label, karena efeknya terbawa antar split.
"""
from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import polars as pl

BASE = Path(os.environ.get("DATAVIDIA_BASE", r"E:\Kuliyah\Datavidia\Task 1"))
DATA = BASE / "datasets"
CACHE = BASE / "exp" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

TARGET = "status_penanganan"
CLASSES = ["normal", "konfirmasi_alamat", "tertunda_operasional", "gagal_antar_awal",
           "retur_pengirim", "rusak_perjalanan", "klaim_hilang"]
K = len(CLASSES)
EVENTS = ["pickup", "sortasi", "transit", "keluar_hub", "upaya_antar", "selesai"]
DEVICES = ["scanner", "pda", "manual", "gerbang_otomatis"]
MODAS = ["darat", "kereta", "laut", "udara"]
INV_TYPES = ["pengiriman", "langganan", "klaim"]
PEAK_MONTHS = [3, 10, 11, 12]
DIMS = ("customers", "couriers", "hubs", "service_packages", "promotions")
ID_COLS = ["shipment_id", "customer_id", "courier_id", "hub_id", "package_id", "promo_id"]
NOISE_COLS = ["skor_prioritas_kirim", "indeks_rute_sekunder", "faktor_muatan_internal",
              "varian_cuaca_rute", "indeks_waktu_proses", "skor_kualitas_cek", "kota_asal", "kota_tujuan"]
CAT_COLS = ["channel", "pickup_type", "goods_category", "dest_region", "kota_asal", "kota_tujuan",
            "region", "segment", "preferred_payment", "loyalty_tier", "kota_tipe", "metode_notifikasi",
            "hub_region", "hub_type", "vehicle_type", "contract_type", "shift_code", "speed_class",
            "handling_class", "tipe_promo", "nama_promo", "rt_moda_first", "rt_moda_last", "inv_pm_first"]

# Sinyal per pengiriman yang diringkas per pelanggan di dalam split.
CS_SIGNALS = [
    "inv_n", "inv_has_klaim", "inv_has_pengiriman", "inv_has_langganan", "inv_amt_sum", "inv_amt_klaim", "inv_disc_sum", "inv_tax_sum",
    "inv_dtp_max", "inv_dtp_mean", "inv_dtp_min", "inv_dtp15_n", "inv_dtp_klaim", "inv_st_gagal", "inv_st_tunda", "inv_penalty", "inv_pm_cod",
    "klaim_to_value",
    "sc_n", "sc_n_upaya_antar", "sc_n_selesai", "sc_n_pickup", "sc_n_transit", "sc_UU", "sc_repickup", "sc_last_selesai", "sc_gap_sum", "sc_lgap_mean",
    "sc_gap_max", "sc_q_mean", "sc_q_min", "sc_q_low_n", "sc_man_n", "sc_man_rate", "sc_gap_U_sum",
    "rt_n", "rt_dur_sum", "rt_lratio_mean", "rt_lratio_max", "rt_dur_null", "rt_cost_sum",
    "is_cod", "is_promo", "jumlah_catatan", "flag_alamat_baru", "log_value", "log_km", "weight_kg", "fragility_flag", "insurance_opt",
    "is_night", "is_peak_month", "is_weekend", "out_of_hours", "order_to_ship_hours", "qty_items", "coupon_count",
]
CS_MAX = ["inv_dtp_max", "inv_amt_klaim", "sc_n_upaya_antar", "sc_gap_max", "jumlah_catatan", "log_value", "rt_lratio_max", "sc_q_low_n"]
CS_MIN = ["inv_dtp_min", "sc_q_min", "sc_q_mean", "rt_lratio_min"]

COURIER_SIGNALS = ["sc_q_mean", "sc_q_min", "sc_man_rate", "sc_lgap_mean", "sc_gap_sum", "sc_n", "sc_n_upaya_antar", "sc_last_selesai",
                   "rt_lratio_mean", "rt_lratio_max", "rt_dur_null", "inv_dtp_mean", "is_night", "is_peak_month", "log_value", "log_km"]
HUB_SIGNALS = ["sc_lgap_mean", "sc_gap_sum", "sc_q_mean", "sc_man_rate", "sc_n", "rt_lratio_mean", "rt_dur_null", "inv_dtp_mean", "inv_late15",
               "is_night", "log_value", "log_km", "sc_n_upaya_antar"]


def scan(split: str, name: str) -> pl.LazyFrame:
    if name in DIMS:
        split = "train"
    return pl.scan_csv(DATA / split / f"{name}.csv", infer_schema_length=200_000)


def safe_div(a, b):
    return pl.when(b.is_not_null() & (b != 0)).then(a / b).otherwise(None)


def fe_scan(lf: pl.LazyFrame) -> pl.LazyFrame:
    lf = lf.select("shipment_id", "event_type", "gap_hours", "device_type", "scan_quality", "is_manual")
    lf = lf.with_columns(pl.col("gap_hours").log1p().alias("lg"))
    e, g, lg, q, m, d = [pl.col(c) for c in ("event_type", "gap_hours", "lg", "scan_quality", "is_manual", "device_type")]
    isu = e == "upaya_antar"
    cumu = isu.cast(pl.Int32).cum_sum()
    aggs = [pl.len().alias("sc_n")]
    aggs += [(e == ev).sum().alias(f"sc_n_{ev}") for ev in EVENTS]
    aggs += [g.sum().alias("sc_gap_sum"), g.mean().alias("sc_gap_mean"), g.median().alias("sc_gap_med"),
             g.max().alias("sc_gap_max"), g.min().alias("sc_gap_min"), g.std().alias("sc_gap_std"),
             lg.mean().alias("sc_lgap_mean"), lg.std().alias("sc_lgap_std"),
             g.first().alias("sc_gap_first"), g.last().alias("sc_gap_last"), g.arg_max().alias("sc_gap_argmax"),
             q.mean().alias("sc_q_mean"), q.min().alias("sc_q_min"), q.max().alias("sc_q_max"), q.std().alias("sc_q_std"),
             q.first().alias("sc_q_first"), q.last().alias("sc_q_last"),
             (q <= 2).sum().alias("sc_q_low_n"), (q == 5).sum().alias("sc_q_top_n"),
             m.sum().alias("sc_man_n"), m.mean().alias("sc_man_rate"),
             (e.last() == "selesai").cast(pl.Int8).alias("sc_last_selesai"),
             (isu & isu.shift(1).fill_null(False)).sum().alias("sc_UU"),
             ((e == "pickup") & (cumu.shift(1).fill_null(0) > 0)).sum().alias("sc_repickup"),
             g.filter(isu).sum().alias("sc_gap_U_sum"), g.filter(isu).max().alias("sc_gap_U_max"),
             q.filter(isu).mean().alias("sc_q_U_mean"),
             g.filter(cumu == 0).sum().alias("sc_gap_before_U")]
    aggs += [(d == x).mean().alias(f"sc_dev_{x}") for x in DEVICES]
    return lf.group_by("shipment_id").agg(aggs)


def nominal_speeds() -> dict:
    r = pl.concat([scan(s, "route_segments").select("moda", "jarak_segmen_km", "durasi_jam") for s in ("train", "test")])
    out = (r.filter(pl.col("durasi_jam") > 0).with_columns((pl.col("jarak_segmen_km") / pl.col("durasi_jam")).alias("v"))
           .group_by("moda").agg(pl.col("v").median()).collect(engine="streaming"))
    return dict(zip(out["moda"].to_list(), out["v"].to_list()))


def fe_route(lf: pl.LazyFrame, vmoda: dict) -> pl.LazyFrame:
    lf = lf.select("shipment_id", "urutan", "moda", "jarak_segmen_km", "durasi_jam", "biaya_segmen").sort("shipment_id", "urutan")
    lf = lf.with_columns(pl.col("moda").replace_strict(vmoda, return_dtype=pl.Float64).alias("v"))
    lf = lf.with_columns((pl.col("jarak_segmen_km") / pl.col("v")).alias("exp_dur"))
    lf = lf.with_columns(
        pl.when(pl.col("durasi_jam") > 0).then((pl.col("durasi_jam") / pl.col("exp_dur")).log()).otherwise(None).alias("lratio"),
        pl.when(pl.col("durasi_jam").is_not_null()).then(pl.col("exp_dur")).otherwise(None).alias("exp_dur_ok"))
    km, dur, cost, md, lr = [pl.col(c) for c in ("jarak_segmen_km", "durasi_jam", "biaya_segmen", "moda", "lratio")]
    aggs = [pl.len().alias("rt_n"), md.n_unique().alias("rt_n_moda"), km.max().alias("rt_km_max"),
            dur.sum().alias("rt_dur_sum"), dur.max().alias("rt_dur_max"), dur.null_count().alias("rt_dur_null"),
            pl.col("exp_dur_ok").sum().alias("rt_exp_dur"), lr.mean().alias("rt_lratio_mean"), lr.max().alias("rt_lratio_max"),
            lr.min().alias("rt_lratio_min"), lr.std().alias("rt_lratio_std"), cost.sum().alias("rt_cost_sum"),
            md.first().alias("rt_moda_first"), md.last().alias("rt_moda_last")]
    for x in MODAS:
        aggs += [(md == x).sum().alias(f"rt_n_{x}"), km.filter(md == x).sum().alias(f"rt_km_{x}")]
    return lf.group_by("shipment_id").agg(aggs)


def fe_invoice(lf: pl.LazyFrame) -> pl.LazyFrame:
    t, amt, dsc, tax, dtp, st, pen, im, pm = [pl.col(c) for c in ("invoice_type", "amount", "discount", "tax", "days_to_pay",
                                                                  "payment_status", "is_penalty", "issued_month", "payment_method")]
    aggs = [pl.len().alias("inv_n"), amt.sum().alias("inv_amt_sum"), dsc.sum().alias("inv_disc_sum"), tax.sum().alias("inv_tax_sum"),
            dtp.max().alias("inv_dtp_max"), dtp.mean().alias("inv_dtp_mean"), dtp.min().alias("inv_dtp_min"), (dtp >= 15).sum().alias("inv_dtp15_n"),
            (st == "gagal").sum().alias("inv_st_gagal"), (st == "tertunda").sum().alias("inv_st_tunda"), pen.max().alias("inv_penalty"),
            im.max().alias("inv_month_max"), im.min().alias("inv_month_min"), (pm == "cod").sum().alias("inv_pm_cod"), pm.first().alias("inv_pm_first")]
    for x in INV_TYPES:
        k = t == x
        aggs += [k.sum().alias(f"inv_has_{x}"), amt.filter(k).sum().alias(f"inv_amt_{x}"), dtp.filter(k).max().alias(f"inv_dtp_{x}"),
                 im.filter(k).max().alias(f"inv_month_{x}")]
    return lf.group_by("shipment_id").agg(aggs)


def join_dims(ship: pl.LazyFrame) -> pl.LazyFrame:
    cu = scan("train", "customers")
    co = scan("train", "couriers").drop("hub_id")
    hb = scan("train", "hubs").rename({"region": "hub_region"})
    pk = scan("train", "service_packages")
    pr = scan("train", "promotions")
    return (ship.join(cu, on="customer_id", how="left").join(co, on="courier_id", how="left").join(hb, on="hub_id", how="left")
            .join(pk, on="package_id", how="left").join(pr, on="promo_id", how="left"))


def fe_derived(lf: pl.LazyFrame) -> pl.LazyFrame:
    c = pl.col
    lf = lf.with_columns(
        c("declared_value").log1p().alias("log_value"),
        c("distance_km").log1p().alias("log_km"),
        safe_div(c("declared_value"), c("weight_kg")).log1p().alias("log_value_per_kg"),
        safe_div(c("weight_kg"), c("max_weight_kg")).alias("w_over_pkg"),
        safe_div(c("weight_kg"), c("kapasitas_muatan_kg")).alias("w_over_vehicle"),
        pl.max_horizontal("weight_kg", "berat_volumetrik").alias("chargeable_kg"),
        safe_div(c("volume_l"), c("weight_kg")).alias("vol_per_kg"),
        c("ship_month").is_in(PEAK_MONTHS).cast(pl.Int8).alias("is_peak_month"),
        (c("ship_dow") >= 5).cast(pl.Int8).alias("is_weekend"),
        ((c("ship_hour") < 6) | (c("ship_hour") >= 20)).cast(pl.Int8).alias("is_night"),
        ((c("ship_hour") < c("jam_buka")) | (c("ship_hour") >= c("jam_tutup"))).cast(pl.Int8).alias("out_of_hours"),
        (c("jam_tutup") - c("jam_buka")).alias("hub_open_hours"),
        (c("ship_hour") - c("jam_buka")).alias("hours_since_open"),
        (c("jam_tutup") - c("ship_hour")).alias("hours_to_close"),
        (c("promo_id") != "TANPA_PROMO").cast(pl.Int8).alias("has_promo"),
        ((c("is_promo") == 1) & (((c("bulan_mulai") <= c("bulan_akhir")) & (c("ship_month") >= c("bulan_mulai")) & (c("ship_month") <= c("bulan_akhir")))
                                 | ((c("bulan_mulai") > c("bulan_akhir")) & ((c("ship_month") >= c("bulan_mulai")) | (c("ship_month") <= c("bulan_akhir")))))).cast(pl.Int8).alias("promo_in_window"),
        (c("region") == c("dest_region")).cast(pl.Int8).alias("dest_is_home"),
        (c("hub_region") == c("dest_region")).cast(pl.Int8).alias("dest_is_hub_region"),
        (c("estimasi_hari") * 24).alias("sla_hours"),
        safe_div(c("rt_dur_sum"), c("rt_exp_dur")).alias("rt_delay_ratio"),
        safe_div(c("distance_km"), c("rt_dur_sum")).alias("rt_speed"),
        safe_div(c("rt_cost_sum"), c("distance_km")).alias("rt_cost_per_km"),
        safe_div(c("rt_km_max"), c("distance_km")).alias("rt_longest_share"),
        ((c("inv_month_max") - c("ship_month")) % 12).alias("inv_month_lag"),
        ((c("inv_month_klaim") - c("ship_month")) % 12).alias("klaim_month_lag"),
        safe_div(c("inv_amt_klaim").cast(pl.Float64), c("declared_value")).alias("klaim_to_value"),
        safe_div(c("inv_disc_sum"), c("inv_amt_sum")).alias("inv_disc_rate"),
        (c("inv_dtp_max") >= 15).cast(pl.Int8).alias("inv_late15"),
        safe_div(c("sc_n_upaya_antar").cast(pl.Float64), c("sc_n")).alias("sc_share_U"),
        safe_div(c("sc_gap_before_U"), c("sc_gap_sum")).alias("sc_share_before_U"),
        safe_div(c("sc_gap_max"), c("sc_gap_sum")).alias("sc_gap_max_share"),
    )
    return lf.with_columns(
        safe_div(c("sc_gap_sum"), c("sla_hours")).alias("scan_vs_sla"),
        safe_div(c("rt_dur_sum"), c("sla_hours")).alias("route_vs_sla"),
        (c("is_peak_month") * c("density_index")).alias("peak_x_density"),
        (c("density_index") * c("capacity_index")).alias("density_x_capacity"),
        (c("fragile_score") * c("log_km")).alias("fragile_x_km"),
        (c("fragile_score") * c("rt_n")).alias("fragile_x_segments"),
        (c("log_value") * (6 - c("automation_level"))).alias("value_x_lowauto"),
        ((6 - c("address_quality")) * c("complaint_index")).alias("addr_x_complaint"),
        (c("is_cod") * c("complaint_index")).alias("cod_x_complaint"),
        ((6 - c("address_quality")) * (c("jumlah_catatan") > 0).cast(pl.Int8)).alias("addr_x_notes"),
        ((6 - c("address_quality")) * c("flag_alamat_baru")).alias("addr_x_newaddr"),
        ((6 - c("address_quality")) * c("sc_n_upaya_antar")).alias("addr_x_attempts"),
    )


def add_customer_within_split(df: pl.DataFrame) -> pl.DataFrame:
    """Agregat pelanggan di dalam split: rata-rata semua baris dan rata-rata saudara (tanpa baris sendiri)."""
    sig = [c for c in CS_SIGNALS if c in df.columns]
    g = df.group_by("customer_id").agg(
        pl.len().cast(pl.Int32).alias("cs_n"),
        *[pl.col(c).cast(pl.Float64).fill_null(0).sum().alias(f"__s_{c}") for c in sig],
        *[pl.col(c).cast(pl.Float64).max().cast(pl.Float32).alias(f"cs_max_{c}") for c in CS_MAX],
        *[pl.col(c).cast(pl.Float64).min().cast(pl.Float32).alias(f"cs_min_{c}") for c in CS_MIN],
    )
    df = df.join(g, on="customer_id", how="left")
    ex = []
    for c in sig:
        v = pl.col(c).cast(pl.Float64).fill_null(0)
        ex.append((pl.col(f"__s_{c}") / pl.col("cs_n")).cast(pl.Float32).alias(f"cs_mean_{c}"))
        ex.append(pl.when(pl.col("cs_n") > 1).then((pl.col(f"__s_{c}") - v) / (pl.col("cs_n") - 1)).otherwise(None).cast(pl.Float32).alias(f"cs_loo_{c}"))
    for c in ("inv_has_klaim", "inv_st_gagal", "inv_st_tunda", "inv_penalty", "inv_dtp15_n"):
        ex.append(pl.col(f"__s_{c}").cast(pl.Float32).alias(f"cs_sum_{c}"))
    return df.with_columns(ex).drop([f"__s_{c}" for c in sig])


def build_split(split: str, vmoda: dict, verbose=True) -> pl.DataFrame:
    t0 = time.time()
    sc = fe_scan(scan(split, "scan_events")).collect()
    rt = fe_route(scan(split, "route_segments"), vmoda).collect()
    iv = fe_invoice(scan(split, "invoices")).collect()
    df = join_dims(scan(split, "shipments")).collect()
    df = df.join(sc, on="shipment_id", how="left").join(rt, on="shipment_id", how="left").join(iv, on="shipment_id", how="left")
    del sc, rt, iv
    gc.collect()
    df = fe_derived(df.lazy()).collect()
    df = add_customer_within_split(df)
    if verbose:
        print(f"  {split}: {df.shape} ({time.time() - t0:.0f}s)", flush=True)
    return df


def build_all(force=False, verbose=True):
    ptr, pte, pme = CACHE / "fe_train.parquet", CACHE / "fe_test.parquet", CACHE / "fe_meta.json"
    if not force and ptr.exists() and pte.exists() and pme.exists():
        return pl.read_parquet(ptr), pl.read_parquet(pte), json.loads(pme.read_text())
    t0 = time.time()
    vmoda = nominal_speeds()
    frames = {s: build_split(s, vmoda, verbose) for s in ("train", "test")}
    keys = ["shipment_id", "courier_id", "hub_id", "ship_month"]
    need = sorted(set(COURIER_SIGNALS) | set(HUB_SIGNALS))
    allv = pl.concat([frames[s].select(keys + need) for s in ("train", "test")])
    cr = allv.group_by("courier_id").agg([pl.len().alias("cr_n")] + [pl.col(c).cast(pl.Float64).mean().cast(pl.Float32).alias(f"cr_{c}") for c in COURIER_SIGNALS])
    hb = allv.group_by("hub_id").agg([pl.len().alias("hb_n")] + [pl.col(c).cast(pl.Float64).mean().cast(pl.Float32).alias(f"hb_{c}") for c in HUB_SIGNALS])
    hm = allv.group_by("hub_id", "ship_month").agg(pl.len().alias("hm_n"), pl.col("sc_lgap_mean").mean().cast(pl.Float32).alias("hm_lgap"),
                                                   pl.col("rt_lratio_mean").mean().cast(pl.Float32).alias("hm_lratio"), pl.col("sc_q_mean").mean().cast(pl.Float32).alias("hm_q"))
    cm = allv.group_by("courier_id", "ship_month").agg(pl.len().alias("cm_n"), pl.col("sc_lgap_mean").mean().cast(pl.Float32).alias("cm_lgap"),
                                                       pl.col("sc_q_mean").mean().cast(pl.Float32).alias("cm_q"))
    del allv
    gc.collect()
    cat_maps = {c: sorted(x for x in set(frames["train"][c].unique().to_list()) | set(frames["test"][c].unique().to_list()) if x is not None) for c in CAT_COLS}
    for s in ("train", "test"):
        df = frames.pop(s)
        df = df.join(cr, on="courier_id", how="left").join(hb, on="hub_id", how="left")
        df = df.join(hm, on=["hub_id", "ship_month"], how="left").join(cm, on=["courier_id", "ship_month"], how="left")
        df = df.with_columns(
            (pl.col("sc_q_mean") - pl.col("cr_sc_q_mean")).alias("q_vs_courier"),
            (pl.col("sc_man_rate") - pl.col("cr_sc_man_rate")).alias("man_vs_courier"),
            (pl.col("sc_lgap_mean") - pl.col("cr_sc_lgap_mean")).alias("lgap_vs_courier"),
            (pl.col("sc_lgap_mean") - pl.col("hb_sc_lgap_mean")).alias("lgap_vs_hub"),
            (pl.col("sc_lgap_mean") - pl.col("hm_lgap")).alias("lgap_vs_hubmonth"),
            (pl.col("rt_lratio_mean") - pl.col("hb_rt_lratio_mean")).alias("lratio_vs_hub"),
            (pl.col("sc_n_upaya_antar") - pl.col("cs_mean_sc_n_upaya_antar")).alias("attempts_vs_customer"),
            (pl.col("sc_q_mean") - pl.col("cs_mean_sc_q_mean")).alias("q_vs_customer"),
        )
        df = df.with_columns([pl.col(c).replace_strict(m, list(range(len(m))), default=-1, return_dtype=pl.Int16).alias(c) for c, m in cat_maps.items()])
        df = df.drop("bulan_mulai", "bulan_akhir")
        num = [c for c, t in df.schema.items() if t.is_numeric() and c not in CAT_COLS and t != pl.Float32]
        df = df.with_columns([pl.col(c).cast(pl.Float32) for c in num])
        df = df.with_columns((pl.col("hub_id") + "_" + pl.col("ship_month").cast(pl.Int8).cast(pl.String)).alias("hub_month"),
                             (pl.col("courier_id") + "_" + pl.col("ship_month").cast(pl.Int8).cast(pl.String)).alias("courier_month"))
        df = df.sort("shipment_id")
        df.write_parquet(ptr if s == "train" else pte, compression="zstd", compression_level=3)
        if verbose:
            print(f"  simpan {s}: {df.shape}", flush=True)
        del df
        gc.collect()
    meta = {"vmoda": vmoda, "cat_maps": cat_maps}
    pme.write_text(json.dumps(meta, ensure_ascii=False))
    if verbose:
        print(f"feature store selesai ({time.time() - t0:.0f}s)", flush=True)
    return pl.read_parquet(ptr), pl.read_parquet(pte), meta


def customer_folds(customer_ids: np.ndarray, n_folds=5, seed=42) -> np.ndarray:
    """Fold berkelompok per pelanggan: semua baris satu pelanggan berada di fold yang sama."""
    uniq, inv = np.unique(customer_ids, return_inverse=True)
    rng = np.random.RandomState(seed)
    return (rng.permutation(len(uniq)) % n_folds)[inv].astype(np.int8)


if __name__ == "__main__":
    tr, te, _ = build_all(force=True)
    print(tr.shape, te.shape)
