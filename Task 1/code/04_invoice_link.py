"""Mekanisme tagihan: apakah sinyal tagihan melekat pada baris atau pada pelanggan?"""
from pathlib import Path

import numpy as np
import polars as pl

BASE = Path(r"E:\Kuliyah\Datavidia\Task 1")
D = BASE / "datasets"
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(18); pl.Config.set_tbl_width_chars(190); pl.Config.set_float_precision(4)
CLS = ["normal", "konfirmasi_alamat", "tertunda_operasional", "gagal_antar_awal", "retur_pengirim", "rusak_perjalanan", "klaim_hilang"]

s = pl.read_csv(D / "train" / "shipments.csv", columns=["shipment_id", "customer_id", "ship_month", "declared_value", "distance_km", "weight_kg", "package_id", "status_penanganan"])
iv = pl.read_csv(D / "train" / "invoices.csv", schema_overrides={"discount": pl.Float64})
pk = pl.read_csv(D / "train" / "service_packages.csv", columns=["package_id", "biaya_dasar", "cost_multiplier"])
n = s.group_by("customer_id").agg(pl.len().alias("n"))
s = s.join(n, on="customer_id").join(pk, on="package_id", how="left")

print("=== A. Jumlah tagihan per pengiriman menurut tipe ===")
print(iv.group_by("shipment_id").agg((pl.col("invoice_type") == "klaim").sum().alias("k"), (pl.col("invoice_type") == "pengiriman").sum().alias("p"), (pl.col("invoice_type") == "langganan").sum().alias("l"))
      .group_by("k", "p", "l").agg(pl.len().alias("n")).sort("n", descending=True).head(10))

j = iv.join(s.drop("customer_id"), on="shipment_id", how="left")
print("\n=== B. issued_month == ship_month baris tertaut, menurut n baris pelanggan dan tipe tagihan ===")
print(j.with_columns(pl.col("n").clip(upper_bound=6)).group_by("n", "invoice_type").agg((pl.col("issued_month") == pl.col("ship_month")).mean().alias("cocok"), pl.len().alias("baris"))
      .pivot(on="invoice_type", index="n", values="cocok").sort("n"))

print("\n=== C. Korelasi amount dengan atribut baris tertaut, menurut n (tipe pengiriman) ===")
rows = []
for k in [1, 2, 3, 4, 5]:
    m = j.filter((pl.col("invoice_type") == "pengiriman") & (pl.col("n") == k))
    rows.append((k, *[float(np.corrcoef(m["amount"], m[c].cast(pl.Float64).fill_null(0))[0, 1]) for c in ["biaya_dasar", "distance_km", "weight_kg", "declared_value"]]))
print(pl.DataFrame(rows, schema=["n", "biaya_dasar", "distance_km", "weight_kg", "declared_value"], orient="row"))
print("\nKorelasi amount tagihan klaim dengan nilai barang baris tertaut:")
rows = []
for k in [1, 2, 3, 4, 5]:
    m = j.filter((pl.col("invoice_type") == "klaim") & (pl.col("n") == k))
    rows.append((k, m.height, float(np.corrcoef(m["amount"], m["declared_value"])[0, 1]), float((m["amount"] / m["declared_value"]).median())))
print(pl.DataFrame(rows, schema=["n", "baris", "r_amount_declared", "median_rasio"], orient="row"))

# Pelanggan tepat dua baris: apakah klaim di baris sendiri dan di baris saudara setara?
print("\n=== D. Pelanggan 2 baris: P(kelas | klaim baris sendiri, klaim baris saudara) ===")
kl = iv.group_by("shipment_id").agg((pl.col("invoice_type") == "klaim").sum().alias("klaim"), pl.col("days_to_pay").max().alias("dtp"))
two = s.filter(pl.col("n") == 2).join(kl, on="shipment_id", how="left").with_columns(pl.col("klaim").fill_null(0), pl.col("dtp").fill_null(0))
g = two.group_by("customer_id").agg(pl.col("klaim").sum().alias("ks"), pl.col("dtp").sum().alias("ds"))
two = two.join(g, on="customer_id").with_columns((pl.col("ks") - pl.col("klaim")).alias("klaim_sib"), (pl.col("ds") - pl.col("dtp")).alias("dtp_sib"))
t = two.with_columns(pl.col("klaim").clip(upper_bound=1), pl.col("klaim_sib").clip(upper_bound=1)).group_by("klaim", "klaim_sib").agg(
    pl.len().alias("baris"), *[(pl.col("status_penanganan") == c).mean().alias(c[:6]) for c in CLS]).sort("klaim", "klaim_sib")
print(t)

print("\n=== E. Pelanggan 2 baris: tertunda menurut days_to_pay sendiri vs saudara (kelompok tinggi >= 10) ===")
t = two.with_columns((pl.col("dtp") >= 10).alias("dtp_tinggi"), (pl.col("dtp_sib") >= 10).alias("sib_tinggi")).group_by("dtp_tinggi", "sib_tinggi").agg(
    pl.len().alias("baris"), *[(pl.col("status_penanganan") == c).mean().alias(c[:6]) for c in CLS]).sort("dtp_tinggi", "sib_tinggi")
print(t)

print("\n=== F. Pelanggan 2 baris: tabel label bersama (peluang label saudara | label baris) ===")
p = two.select("customer_id", "shipment_id", "status_penanganan")
q = p.join(p, on="customer_id", suffix="_b").filter(pl.col("shipment_id") != pl.col("shipment_id_b"))
c = q.group_by("status_penanganan", "status_penanganan_b").agg(pl.len().alias("x"))
c = c.with_columns((pl.col("x") / pl.col("x").sum().over("status_penanganan")).round(3).alias("p"))
print(c.pivot(on="status_penanganan_b", index="status_penanganan", values="p").select(["status_penanganan"] + CLS).sort("status_penanganan").rename(lambda v: v[:7]))

print("\n=== G. Jumlah klaim pelanggan vs jumlah baris retur+klaim_hilang (pelanggan 3 baris) ===")
three = s.filter(pl.col("n") == 3).join(kl, on="shipment_id", how="left").with_columns(pl.col("klaim").fill_null(0))
print(three.group_by("customer_id").agg(pl.col("klaim").sum().alias("total_klaim"), pl.col("status_penanganan").is_in(["retur_pengirim", "klaim_hilang"]).sum().alias("baris_masalah"))
      .group_by("total_klaim", "baris_masalah").agg(pl.len().alias("n")).pivot(on="baris_masalah", index="total_klaim", values="n").fill_null(0).sort("total_klaim").select(["total_klaim", "0", "1", "2", "3"]))
