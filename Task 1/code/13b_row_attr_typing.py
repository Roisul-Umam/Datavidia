"""Tipe atribut baris pada tabel shipments dengan uji pasangan yang sama seperti 13_signal_typing."""
import sys
import numpy as np
import polars as pl
sys.path.insert(0, r"E:\Kuliyah\Datavidia\Task 1\code")
from features import BASE, CLASSES, K, TARGET, build_all
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(12); pl.Config.set_tbl_width_chars(150); pl.Config.set_float_precision(4)
tr, _, _ = build_all()
cols = ["weight_kg", "volume_l", "declared_value", "log_value", "distance_km", "ship_month", "ship_dow", "ship_hour", "is_cod", "is_promo", "has_promo", "promo_in_window", "insurance_opt", "qty_items", "coupon_count",
        "gift_wrap", "fragility_flag", "order_to_ship_hours", "prioritas_pengiriman", "jumlah_catatan", "flag_alamat_baru", "berat_volumetrik", "is_night", "is_peak_month", "is_weekend", "out_of_hours",
        "peak_x_density", "fragile_x_km", "value_x_lowauto", "addr_x_notes", "addr_x_newaddr", "cod_x_complaint", "w_over_pkg", "w_over_vehicle", "dest_is_home", "dest_is_hub_region", "fragile_score", "automation_level", "density_index"]
static = ["address_quality", "complaint_index", "credit_score", "tenure_months", "is_business", "is_verified", "tagihan_langganan"]
two = tr.filter(pl.col("cs_n") == 2).sort("customer_id").select(["customer_id", TARGET] + list(dict.fromkeys(static + cols)))
n = two.height // 2
Y = np.eye(K)[two[TARGET].replace_strict(CLASSES, list(range(K)), return_dtype=pl.Int8).to_numpy().astype(int)]
Ya, Yb = Y[::2], Y[1::2]
Z = two.select(pl.col(static).cast(pl.Float64)).to_numpy()[::2]
Z = np.where(np.isnan(Z), np.nanmedian(Z, axis=0), Z); Z = (Z - Z.mean(0)) / Z.std(0)
Q, _ = np.linalg.qr(np.hstack([np.ones((n, 1)), Z])); resid = lambda v: v - Q @ (Q.T @ v)
dy, sy = Ya - Yb, resid(Ya + Yb)
rows = []
for c in cols:
    x = two[c].cast(pl.Float64).to_numpy(); x = np.where(np.isnan(x), np.nanmean(x), x); sd = x.std()
    if sd < 1e-9: continue
    x = x / sd; xa, xb = x[::2], x[1::2]; dx, sx = xa - xb, resid(xa + xb)
    b_d, b_s = (dx @ dy) / (dx @ dx), (sx @ sy) / (sx @ sx); own, sib = (b_s + b_d) / 2, (b_s - b_d) / 2
    no, ns = float(np.linalg.norm(own)), float(np.linalg.norm(sib)); share = no**2 / (no**2 + ns**2 + 1e-12); cosv = float(own @ sib / (no * ns + 1e-12))
    kind = "lemah" if max(no, ns) < 0.006 else "sendiri" if share >= .8 else "saudara" if share <= .2 else "rata2" if (.35 <= share <= .65 and cosv >= .8) else "campuran"
    t = int(np.argmax(np.abs(own) + np.abs(sib)))
    rows.append((c, kind, round(no, 4), round(ns, 4), round(share, 3), round(cosv, 3), CLASSES[t][:8], round(float(own[t]), 4), round(float(sib[t]), 4)))
out = pl.DataFrame(rows, schema=["sinyal", "tipe", "kuat_sendiri", "kuat_saudara", "porsi_sendiri", "kos", "kelas_utama", "b_sendiri", "b_saudara"], orient="row")
out = out.with_columns(pl.max_horizontal("kuat_sendiri", "kuat_saudara").alias("k")).sort("k", descending=True).drop("k")
print(out); out.write_csv(BASE / "exp" / "diagnostics" / "13b_row_attr_typing.csv")
