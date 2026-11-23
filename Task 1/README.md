# Prediksi Status Penanganan Pengiriman (Datavidia 11.0)

Model ini menandai satu dari tujuh `status_penanganan` untuk setiap pengiriman. Skor validasi 5-fold (Macro F1) adalah **0,6187**, dengan rentang per fold 0,6178 sampai 0,6198. Skor leaderboard bisa berbeda karena sebaran data test tidak identik dengan train.

## Isi folder

| Berkas | Keterangan |
|---|---|
| `notebook_status_penanganan.ipynb` | Pemrosesan data, EDA, penggabungan tabel, fitur, pelatihan, evaluasi, dan inferensi |
| `models/` | Berkas model yang dimuat saat inferensi (sekitar 215 MB) |
| `submission.csv` | Hasil prediksi untuk 362.245 pengiriman |
| `requirements.txt` | Versi paket yang dipakai |
| `Datasets/` | Data kompetisi |

## Menjalankan inferensi tanpa melatih ulang

1. Pasang Python 3.13.
2. Pasang dependensi:

   ```powershell
   pip install -r requirements.txt
   ```

3. Pastikan `Datasets/` dan `models/` berada dalam satu folder. Data train tetap dibutuhkan karena profil kurir, hub, dan pelanggan dihitung dari train dan test.
4. Buka notebook di Jupyter atau VS Code, lalu jalankan semua sel. Notebook bisa dijalankan dari folder tersebut atau dari subfolder seperti `Code/`.

Seluruh notebook selesai dalam sekitar 8 menit pada CPU 12 thread dengan RAM 16 GB. Hasilnya tersimpan sebagai `submission.csv`. Notebook juga membandingkan prediksinya dengan salinan referensi dari saat pelatihan untuk memastikan berkas model termuat dengan benar.

## Isi folder `models/`

| Berkas | Fungsi |
|---|---|
| `lgbm_fold*.txt`, `xgb_fold*.json`, `cat_fold*.cbm` | 15 model boosting (3 algoritma, 5 fold) |
| `glm_fold*.npz` | Koefisien dan parameter skala regresi logistik |
| `te_*.parquet` | Tabel target encoding pelanggan, kurir, hub, dan hub per bulan |
| `feature_config.json` | Daftar fitur, pemetaan kategori, dan konfigurasi inferensi |
| `training_report.json` | Bobot ensemble, pengali kelas, dan metrik pelatihan |
| `final_weight_cv.json` | Skor validasi untuk aturan keputusan final |
| `oof_predictions.npz`, `reference_test_proba.npy` | Bahan evaluasi dan cek reprodusibilitas |
| `feature_selection.json`, `tuning_result.json`, `drift_diagnostics.json` | Hasil eksperimen yang ditampilkan di notebook |

## Opsi tambahan

Secara default tidak ada proses pelatihan. Variabel lingkungan berikut mengaktifkan proses yang lebih berat dan **menimpa berkas terkait di `models/`**.

| Variabel | Fungsi |
|---|---|
| `RUN_TRAINING=1` | Melatih ulang seluruh ensemble (sekitar 1 jam 45 menit) |
| `RUN_DIAGNOSTICS=1` | Mengulang diagnostik drift (sekitar 6 menit) |
| `RUN_SELECTION=1` | Mengulang seleksi fitur. Daftar fitur di konfigurasi tidak berubah otomatis |
| `RUN_TUNING=1` | Mengulang tuning Optuna. Parameter di konfigurasi tidak berubah otomatis |
| `N_THREADS` | Jumlah thread CPU (default paling banyak 12) |
