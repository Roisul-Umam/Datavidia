"""Tipe tiap sinyal log: milik baris sendiri, riwayat saudara, atau rata-rata pelanggan.

Pada pelanggan tepat dua baris, model peluang linear y_i = a + b_own x_i + b_sib x_j memberi:
  selisih: y_1 - y_2 = (b_own - b_sib)(x_1 - x_2)   -> efek statis pelanggan hilang otomatis
  jumlah : y_1 + y_2 = 2a + (b_own + b_sib)(x_1 + x_2), dengan atribut statis pelanggan dikontrol

Tipe:  sendiri (b_own dominan), saudara (b_sib dominan), rata2 (b_own = b_sib), campuran.
"""
import sys

import numpy as np
import polars as pl

sys.path.insert(0, r"E:\Kuliyah\Datavidia\Task 1\code")
from features import BASE, CAT_COLS, CLASSES, K, TARGET, build_all

pl.Config.set_tbl_rows(200); pl.Config.set_tbl_cols(14); pl.Config.set_tbl_width_chars(170); pl.Config.set_float_precision(4)
tr, te, _ = build_all()
log_cols = [c for c in te.columns if c.startswith(("sc_", "rt_", "inv_")) and c not in CAT_COLS] + \
           ["klaim_to_value", "klaim_month_lag", "inv_month_lag", "inv_disc_rate", "inv_late15", "scan_vs_sla", "route_vs_sla", "rt_delay_ratio", "rt_speed",
            "rt_cost_per_km", "rt_longest_share", "addr_x_attempts"]
log_cols = list(dict.fromkeys(log_cols))
static = ["address_quality", "complaint_index", "credit_score", "tenure_months", "is_business", "is_verified", "tagihan_langganan"]
two = tr.filter(pl.col("cs_n") == 2).sort("customer_id").select(["customer_id", TARGET] + static + log_cols)
assert (two["customer_id"][::2] == two["customer_id"][1::2]).all()
n = two.height // 2
yc = two[TARGET].replace_strict(CLASSES, list(range(K)), return_dtype=pl.Int8).to_numpy().astype(int)
Y = np.eye(K)[yc]
Ya, Yb = Y[::2], Y[1::2]
Z = two.select(pl.col(static).cast(pl.Float64)).to_numpy()[::2]
Z = np.where(np.isnan(Z), np.nanmedian(Z, axis=0), Z)
Z = (Z - Z.mean(0)) / Z.std(0)
Q, _ = np.linalg.qr(np.hstack([np.ones((n, 1)), Z]))
resid = lambda v: v - Q @ (Q.T @ v)
dy, sy = Ya - Yb, resid(Ya + Yb)

rows = []
for c in log_cols:
    x = two[c].cast(pl.Float64).to_numpy()
    if np.isnan(x).mean() > 0.5:
        continue
    x = np.where(np.isnan(x), np.nanmean(x), x)
    sd = x.std()
    if sd < 1e-9:
        continue
    x = x / sd
    xa, xb = x[::2], x[1::2]
    dx, sx = xa - xb, resid(xa + xb)
    b_d, b_s = (dx @ dy) / (dx @ dx), (sx @ sy) / (sx @ sx)
    own, sib = (b_s + b_d) / 2, (b_s - b_d) / 2
    no, ns = float(np.linalg.norm(own)), float(np.linalg.norm(sib))
    share = no ** 2 / (no ** 2 + ns ** 2 + 1e-12)
    cosv = float(own @ sib / (no * ns + 1e-12))
    strength = max(no, ns)
    if strength < 0.006:
        kind = "lemah"
    elif share >= 0.8:
        kind = "sendiri"
    elif share <= 0.2:
        kind = "saudara"
    elif 0.35 <= share <= 0.65 and cosv >= 0.8:
        kind = "rata2"
    else:
        kind = "campuran"
    top = int(np.argmax(np.abs(own) + np.abs(sib)))
    rows.append((c, kind, round(no, 4), round(ns, 4), round(share, 3), round(cosv, 3), CLASSES[top][:8], round(float(own[top]), 4), round(float(sib[top]), 4)))

out = pl.DataFrame(rows, schema=["sinyal", "tipe", "kuat_sendiri", "kuat_saudara", "porsi_sendiri", "kos", "kelas_utama", "b_sendiri", "b_saudara"], orient="row")
out = out.with_columns(pl.max_horizontal("kuat_sendiri", "kuat_saudara").alias("kuat")).sort("kuat", descending=True).drop("kuat")
print(f"pasangan pelanggan dua baris: {n}")
print(out.group_by("tipe").len().sort("len", descending=True))
print(out)
out.write_csv(BASE / "exp" / "diagnostics" / "13_signal_typing.csv")
