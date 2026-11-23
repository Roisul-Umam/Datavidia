"""Mekanisme saudara: apakah saudara berbagi kurir/hub, dan sinyal apa yang tersisa setelah kurir dikontrol."""
import sys
import numpy as np
import polars as pl
sys.path.insert(0, r"E:\Kuliyah\Datavidia\Task 1\code")
from features import BASE, CLASSES, K, TARGET, build_all
pl.Config.set_tbl_rows(70); pl.Config.set_tbl_cols(14); pl.Config.set_tbl_width_chars(170); pl.Config.set_float_precision(4)
tr, te, _ = build_all()
cols = ["shipment_id", "customer_id", "courier_id", "hub_id", "ship_month", "ship_dow", "ship_hour", "sc_q_mean", "sc_man_rate", "sc_lgap_mean", "sc_n_upaya_antar",
        "rt_lratio_mean", "inv_has_klaim", "inv_dtp_max", "cr_sc_q_mean", "cr_sc_man_rate", "cr_sc_lgap_mean", "cs_n"]

print("=== A. Peluang saudara memakai kurir / hub / bulan yang sama ===")
def share(a, b, same):
    j = a.select("shipment_id", "customer_id", "courier_id", "hub_id", "ship_month").join(
        b.select(pl.col("shipment_id").alias("sid_b"), "customer_id", pl.col("courier_id").alias("c_b"), pl.col("hub_id").alias("h_b"), pl.col("ship_month").alias("m_b")), on="customer_id")
    if same: j = j.filter(pl.col("shipment_id") != pl.col("sid_b"))
    return j.height, float((j["courier_id"] == j["c_b"]).mean()), float((j["hub_id"] == j["h_b"]).mean()), float((j["ship_month"] == j["m_b"]).mean())
for nm, a, b, s in [("train-train", tr, tr, True), ("test-test", te, te, True), ("test-train", te, tr, False)]:
    print(nm, "pasangan=%d  kurir_sama=%.4f  hub_sama=%.4f  bulan_sama=%.4f" % share(a, b, s))
print("acuan acak: kurir %.6f | hub %.6f" % (1 / tr["courier_id"].n_unique(), float((tr["hub_id"].value_counts(normalize=True)["proportion"] ** 2).sum())))

print("\n=== B. Pasangan saudara (cs_n = 2): korelasi label baris dengan log saudara, menurut kurir sama/beda ===")
two = tr.filter(pl.col("cs_n") == 2).select(cols + [TARGET])
p = two.join(two.select(pl.col("shipment_id").alias("sid_b"), "customer_id", pl.col("courier_id").alias("c_b"), pl.col("sc_q_mean").alias("q_b"), pl.col("sc_man_rate").alias("man_b"),
                        pl.col("sc_lgap_mean").alias("lgap_b"), pl.col("sc_n_upaya_antar").alias("up_b"), pl.col("inv_has_klaim").alias("kl_b"), pl.col("inv_dtp_max").alias("dtp_b"), pl.col(TARGET).alias("y_b")), on="customer_id").filter(pl.col("shipment_id") != pl.col("sid_b"))
p = p.with_columns((pl.col("courier_id") == pl.col("c_b")).alias("same_courier"))
print("porsi pasangan kurir sama:", round(float(p["same_courier"].mean()), 4))
rows = []
for nm, f in [("kurir sama", True), ("kurir beda", False)]:
    k = p.filter(pl.col("same_courier") == f)
    for c, s, o in [("rusak_perjalanan", "q_b", "sc_q_mean"), ("rusak_perjalanan", "man_b", "sc_man_rate"), ("gagal_antar_awal", "up_b", "sc_n_upaya_antar"), ("konfirmasi_alamat", "up_b", "sc_n_upaya_antar"),
                    ("retur_pengirim", "kl_b", "inv_has_klaim"), ("tertunda_operasional", "dtp_b", "inv_dtp_max"), ("tertunda_operasional", "lgap_b", "sc_lgap_mean")]:
        t = (k[TARGET] == c).cast(pl.Float64).to_numpy()
        rows.append((nm, k.height, c[:10], s, float(np.corrcoef(t, k[o].fill_null(0).to_numpy())[0, 1]), float(np.corrcoef(t, k[s].fill_null(0).to_numpy())[0, 1])))
print(pl.DataFrame(rows, schema=["pasangan", "n", "kelas", "log_saudara", "r_sendiri", "r_saudara"], orient="row"))

print("\n=== C. Kesamaan label antar saudara: kurir sama vs beda ===")
print(p.group_by("same_courier").agg(pl.len().alias("n"), (pl.col(TARGET) == pl.col("y_b")).mean().alias("label_sama")))

print("\n=== D. Rusak: rata-rata q dan manual baris ini menurut (label sendiri rusak?, label saudara rusak?) ===")
print(p.with_columns((pl.col(TARGET) == "rusak_perjalanan").alias("y_rusak"), (pl.col("y_b") == "rusak_perjalanan").alias("sib_rusak")).group_by("y_rusak", "sib_rusak").agg(
    pl.len().alias("n"), pl.col("sc_q_mean").mean().alias("q_sendiri"), pl.col("q_b").mean().alias("q_saudara"), pl.col("sc_man_rate").mean().alias("man_sendiri"), pl.col("man_b").mean().alias("man_saudara")).sort("y_rusak", "sib_rusak"))

print("\n=== E. Sidik jari log per kelas pada baris sendiri (cs_n = 1) ===")
one = tr.filter(pl.col("cs_n") == 1)
print(one.group_by(TARGET).agg(pl.len().alias("n"), pl.col("sc_q_mean").mean(), pl.col("sc_man_rate").mean(), pl.col("sc_lgap_mean").mean(), pl.col("sc_n_upaya_antar").mean(), pl.col("inv_has_klaim").mean(), pl.col("inv_dtp_max").mean(), pl.col("rt_lratio_mean").mean()).sort("n", descending=True))
