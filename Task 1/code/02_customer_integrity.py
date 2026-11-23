"""Mengapa efek pelanggan tidak terbawa ke test? Uji integritas relasi dan koherensi per pelanggan."""
from pathlib import Path

import numpy as np
import polars as pl

BASE = Path(r"E:\Kuliyah\Datavidia\Task 1")
D = BASE / "datasets"
rng = np.random.default_rng(42)
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_cols(16); pl.Config.set_tbl_width_chars(170); pl.Config.set_float_precision(4)

ship_cols = ["shipment_id", "customer_id", "courier_id", "hub_id", "ship_month", "is_cod", "dest_region", "jumlah_catatan", "flag_alamat_baru"]
S = {s: pl.read_csv(D / s / "shipments.csv", columns=ship_cols + (["status_penanganan"] if s == "train" else [])) for s in ("train", "test")}
cu = pl.read_csv(D / "train" / "customers.csv")
co = pl.read_csv(D / "train" / "couriers.csv", columns=["courier_id", "hub_id"]).rename({"hub_id": "courier_hub"})
hb = pl.read_csv(D / "train" / "hubs.csv", columns=["hub_id", "region"]).rename({"region": "hub_region"})

print("=== 1. Integritas relasi antar tabel (porsi baris yang cocok) ===", flush=True)
rows = []
for s in ("train", "test"):
    sh = S[s]
    iv = pl.read_csv(D / s / "invoices.csv", columns=["shipment_id", "customer_id", "payment_method", "issued_month", "invoice_type"], schema_overrides={"customer_id": pl.String})
    j = iv.join(sh.select("shipment_id", pl.col("customer_id").alias("ship_cust"), "ship_month", "is_cod"), on="shipment_id", how="left")
    j = j.join(cu.select("customer_id", "preferred_payment"), left_on="ship_cust", right_on="customer_id", how="left")
    rows.append((s, "invoice.customer_id == shipment.customer_id", float((j["customer_id"] == j["ship_cust"]).mean())))
    rows.append((s, "invoice.payment_method == customer.preferred", float((j["payment_method"] == j["preferred_payment"]).mean())))
    rows.append((s, "invoice.issued_month == ship_month", float((j["issued_month"] == j["ship_month"]).mean())))
    rows.append((s, "invoice.method==cod <=> is_cod", float(((j["payment_method"] == "cod") == (j["is_cod"] == 1)).mean())))
    sc = pl.read_csv(D / s / "scan_events.csv", columns=["shipment_id", "hub_id"]).rename({"hub_id": "scan_hub"}).unique("shipment_id")
    rt = (pl.read_csv(D / s / "route_segments.csv", columns=["shipment_id", "urutan", "segmen_hub_id"]).filter(pl.col("urutan") == 1).drop("urutan"))
    k = sh.join(sc, on="shipment_id", how="left").join(rt, on="shipment_id", how="left").join(co, on="courier_id", how="left")
    k = k.join(cu.select("customer_id", "region"), on="customer_id", how="left").join(hb, on="hub_id", how="left")
    rows.append((s, "scan.hub_id == shipment.hub_id", float((k["scan_hub"] == k["hub_id"]).mean())))
    rows.append((s, "route.seg1_hub == shipment.hub_id", float((k["segmen_hub_id"] == k["hub_id"]).mean())))
    rows.append((s, "courier.hub_id == shipment.hub_id", float((k["courier_hub"] == k["hub_id"]).mean())))
    rows.append((s, "customer.region == hub.region", float((k["region"] == k["hub_region"]).mean())))
    rows.append((s, "customer.region == dest_region", float((k["region"] == k["dest_region"]).mean())))
    rows.append((s, "customer_id ada di customers.csv", float(k["region"].is_not_null().mean())))
t = pl.DataFrame(rows, schema=["split", "relasi", "cocok"], orient="row").pivot(on="split", index="relasi", values="cocok")
print(t)

print("\n=== 2. Atribut pelanggan vs perilaku baris (korelasi; train vs test) ===", flush=True)
def agg(s):
    a = (pl.scan_csv(D / s / "scan_events.csv").group_by("shipment_id").agg((pl.col("event_type") == "upaya_antar").sum().alias("n_upaya"),
         pl.col("scan_quality").mean().alias("q_mean")).collect(engine="streaming"))
    b = (pl.scan_csv(D / s / "invoices.csv", schema_overrides={"discount": pl.Float64}).group_by("shipment_id").agg(
        (pl.col("invoice_type") == "klaim").sum().alias("inv_klaim"), pl.col("days_to_pay").mean().alias("dtp"),
        (pl.col("payment_method") == "cod").sum().alias("inv_cod")).collect(engine="streaming"))
    return S[s].join(a, on="shipment_id").join(b, on="shipment_id", how="left").join(cu, on="customer_id", how="left")

A = {s: agg(s) for s in ("train", "test")}
pairs = [("address_quality", "n_upaya"), ("address_quality", "jumlah_catatan"), ("address_quality", "flag_alamat_baru"), ("complaint_index", "inv_klaim"),
         ("complaint_index", "n_upaya"), ("complaint_index", "dtp"), ("credit_score", "dtp"), ("is_business", "is_cod"), ("tenure_months", "dtp"),
         ("is_verified", "n_upaya"), ("tagihan_langganan", "inv_klaim")]
rows = []
for a, b in pairs:
    r = [np.corrcoef(*A[s].select(pl.col(a).cast(pl.Float64), pl.col(b).cast(pl.Float64)).drop_nulls().to_numpy().T)[0, 1] for s in ("train", "test")]
    rows.append((a, b, r[0], r[1]))
print(pl.DataFrame(rows, schema=["atribut", "perilaku", "r_train", "r_test"], orient="row"))

print("\n=== 3. Koherensi antar-baris dari pelanggan yang sama (korelasi pasangan) ===", flush=True)
def pair_corr(dfa, dfb, col, same=False, max_rows=400_000):
    a = dfa.select("shipment_id", "customer_id", pl.col(col).cast(pl.Float64).alias("x")).drop_nulls()
    b = dfb.select("shipment_id", "customer_id", pl.col(col).cast(pl.Float64).alias("y")).drop_nulls()
    # pilih satu pasangan acak per baris a
    b = b.with_columns(pl.Series("r", rng.random(b.height)))
    j = a.join(b, on="customer_id")
    if same:
        j = j.filter(pl.col("shipment_id") != pl.col("shipment_id_right"))
    j = j.sort("r").unique("shipment_id", keep="first")
    if j.height > max_rows:
        j = j.sample(max_rows, seed=1)
    return float(np.corrcoef(j["x"], j["y"])[0, 1]), j.height

tr, te = A["train"], A["test"]
mask = rng.random(tr.height) < 0.5
tra, trb = tr.filter(pl.Series(mask)), tr.filter(pl.Series(~mask))
rows = []
for c in ["n_upaya", "inv_klaim", "q_mean", "dtp", "is_cod", "jumlah_catatan", "flag_alamat_baru", "inv_cod"]:
    r1, n1 = pair_corr(tra, trb, c)
    r2, n2 = pair_corr(te, te, c, same=True)
    r3, n3 = pair_corr(te, tr, c)
    rows.append((c, r1, r2, r3, n1, n2, n3))
print(pl.DataFrame(rows, schema=["kolom", "train_vs_train", "test_vs_test", "test_vs_train", "n1", "n2", "n3"], orient="row"))

print("\n=== 4. Kesepakatan label antar baris pelanggan menurut jarak bulan (train) ===", flush=True)
d = S["train"].select("shipment_id", "customer_id", "ship_month", "status_penanganan")
p = d.join(d, on="customer_id", suffix="_b").filter(pl.col("shipment_id") < pl.col("shipment_id_b"))
p = p.with_columns((pl.col("ship_month") - pl.col("ship_month_b")).abs().alias("dm"))
p = p.with_columns(pl.min_horizontal("dm", 12 - pl.col("dm")).alias("dm_circ"),
                   (pl.col("status_penanganan") == pl.col("status_penanganan_b")).alias("same"))
print(p.group_by("dm_circ").agg(pl.len().alias("pasangan"), pl.col("same").mean().alias("label_sama")).sort("dm_circ"))
base = float((S["train"]["status_penanganan"].value_counts(normalize=True)["proportion"] ** 2).sum())
print("peluang label sama jika independen:", round(base, 4))
print("\nlabel_sama menurut kelas:")
print(p.group_by("status_penanganan").agg(pl.col("same").mean().alias("p_sama"), pl.len().alias("n")).sort("n", descending=True))

print("\n=== 5. Jumlah baris per pelanggan: train vs test ===", flush=True)
ct = S["train"].group_by("customer_id").agg(pl.len().alias("n_tr")).join(S["test"].group_by("customer_id").agg(pl.len().alias("n_te")), on="customer_id", how="full", coalesce=True).fill_null(0)
ct = ct.with_columns((pl.col("n_tr") + pl.col("n_te")).alias("n_all"), (pl.col("n_te") / (pl.col("n_tr") + pl.col("n_te"))).alias("share_test"))
print(ct.group_by(pl.col("n_all").clip(upper_bound=10)).agg(pl.len().alias("pelanggan"), pl.col("share_test").mean().alias("porsi_test_mean"),
      (pl.col("share_test") == 0).mean().alias("tanpa_test"), (pl.col("share_test") == 1).mean().alias("semua_test")).sort("n_all"))
