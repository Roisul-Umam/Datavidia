"""Apakah log sebuah baris dibangkitkan dari status baris saudara tertentu (misalnya baris sebelumnya)?

Untuk pelanggan dengan n baris, baris diurutkan menurut suatu kunci, lalu dihitung matriks korelasi
r(status posisi a, log posisi b). Pola tak simetris menunjukkan urutan yang dipakai generator.
"""
import sys
import numpy as np
import polars as pl
sys.path.insert(0, r"E:\Kuliyah\Datavidia\Task 1\code")
from features import BASE, DATA, TARGET, build_all
np.set_printoptions(precision=3, suppress=True, linewidth=150)
tr, _, _ = build_all()
order = pl.read_csv(DATA / "train" / "shipments.csv", columns=["shipment_id"]).with_row_index("pos")
d = tr.select("shipment_id", "customer_id", "cs_n", "ship_month", "ship_dow", "ship_hour", "order_to_ship_hours", TARGET, "sc_n_upaya_antar", "sc_man_rate", "sc_q_mean", "inv_has_klaim", "inv_dtp_max", "sc_lgap_mean").join(order, on="shipment_id")
PAIRS = [("gagal_antar_awal", "sc_n_upaya_antar"), ("konfirmasi_alamat", "sc_n_upaya_antar"), ("rusak_perjalanan", "sc_man_rate"), ("rusak_perjalanan", "sc_q_mean"),
         ("retur_pengirim", "inv_has_klaim"), ("tertunda_operasional", "inv_dtp_max"), ("tertunda_operasional", "sc_lgap_mean")]
KEYS = {"urutan file": ["pos"], "shipment_id": ["shipment_id"], "waktu (bulan,hari,jam)": ["ship_month", "ship_dow", "ship_hour", "pos"]}

def matrix(n, key, cls, col):
    g = d.filter(pl.col("cs_n") == n).sort(["customer_id"] + key).with_columns(pl.int_range(pl.len()).over("customer_id").alias("p"))
    y = (g[TARGET] == cls).cast(pl.Float64).to_numpy().reshape(-1, n)
    x = g[col].cast(pl.Float64).fill_null(0).to_numpy().reshape(-1, n)
    return np.array([[np.corrcoef(y[:, a], x[:, b])[0, 1] for b in range(n)] for a in range(n)])

for kname, key in KEYS.items():
    print(f"\n########## Urutan menurut: {kname} ##########")
    for n in (2, 3):
        print(f"--- pelanggan {n} baris (baris = posisi status, kolom = posisi log) ---")
        for cls, col in PAIRS:
            m = matrix(n, key, cls, col)
            print(f"{cls[:10]:10s} ~ {col[:14]:14s}", " | ".join(" ".join(f"{v:+.3f}" for v in row) for row in m))
