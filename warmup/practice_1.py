import numpy as np
import pandas as pd

# 1. Buat data harian: pola musim + efek akhir pekan + noise
rng = np.random.default_rng(42)
tanggal = pd.date_range("2022-01-01", "2024-12-31", freq="D")
musim = 100 + 30 * np.sin(2 * np.pi * tanggal.dayofyear.to_numpy() / 365.25)
akhir_pekan = 15 * (tanggal.dayofweek >= 5)
noise = rng.normal(0, 12, len(tanggal))
df = pd.DataFrame({"penjualan": musim + akhir_pekan + noise}, index=tanggal)

# 2. Masa lalu untuk latihan, masa depan untuk ujian
latih = df.loc[:"2024-09-30", "penjualan"]
uji = df.loc["2024-10-01":, "penjualan"]

# 3. Metrik: rata-rata besar kesalahan
def mae(aktual, prediksi):
    return np.mean(np.abs(aktual - prediksi))

# 4. Tebakan paling sederhana sebagai pembanding
baseline = {
    "rata-rata semua data": latih.mean(),
    "median semua data": latih.median(),
    "median 28 hari terakhir": latih.iloc[-28:].median(),
}
for nama, prediksi in baseline.items():
    print(f"{nama}: MAE {mae(uji, prediksi):.2f}")