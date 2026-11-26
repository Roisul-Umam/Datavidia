# Klasifikasi Jenis Fenomena dari Pasangan Citra Udara (Datavidia 11.0, Task 2)

Setiap `Id` adalah sepasang citra 1024x1024 (`pre.jpg`, `post.jpg`) yang harus diberi satu dari enam kelas: `Earthquake`, `Fire`, `Hurricane`, `Tsunami`, `Volcano`, `Explosion`. Metriknya Macro F1.

**Data ini tidak punya label latih.** Ke-3.180 pasangan di `Datasets/Fenomena/` adalah data yang harus diprediksi, jadi solusi ini tidak melatih classifier dengan label. Prediksi dihasilkan oleh model pralatih secara zero-shot, lalu disatukan oleh label model tanpa label. Satu-satunya informasi label yang dipakai berasal dari halaman Evaluation: submission berisi `Earthquake` semua mendapat skor ~0,06, sehingga porsi Earthquake ≈ 22%. Tidak ada prediksi yang diisi tangan.

Skor leaderboard belum diketahui. Karena tidak ada label, akurasi tidak bisa diukur secara lokal. Validasi memakai pemeriksaan tanpa label (bagian 7 notebook).

## Isi folder

| Berkas | Keterangan |
|---|---|
| `notebook_task2_fenomena.ipynb` | EDA, reduksi noise, representasi, bukti zero-shot, label model, validasi tanpa label, dan inferensi |
| `submission.csv` | Prediksi untuk 3.180 pasangan, identik dengan `exp/submissions/submission_final.csv` |
| `models/label_model_final.npz`, `.json` | Parameter label model: rata-rata dan skala kalibrasi tiap sinyal, koefisien deconfounding, bias kelas, bobot fusi, konfigurasi pipeline |
| `code/` | Skrip pipeline (`t2pipe.py` dipanggil notebook) dan skrip eksperimen bernomor |
| `requirements.txt` | Versi paket |
| `exp/` | Cache fitur (`exp/cache/`, tidak di-commit), gambar EDA (`exp/viz/`), log, submission per versi |

## Pipeline

1. **Reduksi noise.** Hasil survei pada 6.360 citra: padding hitam akibat warp (84% citra), salt-and-pepper (36%), stripe horizontal (38%), awan sintetis (9% citra dengan awan > 5%), grayscale (7,5% pasangan), serta warna bergeser atau kanal tertukar. Penanganannya: mask padding lewat flood fill dari tepi; median 3x3 hanya pada piksel yang menyimpang; destriping per baris (skor stripe median turun dari 2,19 ke 0,94, p90 dari 9,41 ke 1,70); downsample ke 512; stretch persentil per kanal; detektor awan v2 yang hanya menangkap blob besar bertepi lembut (v1 salah mendeteksi atap putih dan salju).
2. **Representasi.** DINOv2-L (with registers) dan CLIP ViT-L/14-336 pada versi warna dan grayscale, dirata-rata atas 4 rotasi, dengan pooling hanya pada patch valid tanpa awan. Uji tanpa label: dari citra post, DINOv2 menemukan citra pre pasangannya di antara 3.180 kandidat dengan R@1 0,46. Struktur event memakai fitur grayscale supaya pasangan grayscale tidak membentuk klaster sendiri.
3. **Bukti zero-shot.**
   - Qwen3-VL-8B (4-bit) membaca citra berwarna, Qwen3-VL-4B membaca citra grayscale. Keduanya menjawab soal pilihan ganda before/after; jawaban diambil dari logit huruf, dirata-rata atas 3 urutan opsi.
   - Skor perubahan CLIP: kemiripan dengan teks setelah bencana dikurangi sebelumnya.
4. **Label model.**
   - **Kalibrasi per kelas.** Bias global VLM besar: 4B condong ke Volcano, 8B ke Fire.
   - **Deconfounding.** Korupsi diterapkan acak, jadi ketergantungan skor pada luas awan, grayscale, stripe dan sejenisnya adalah bias dan diregresikan keluar. Contohnya, sisa awan terbaca sebagai asap ledakan.
   - **Fusi dan propagasi.** Sinyal digabung, lalu dipropagasi di graf kNN event (α = 0,95). Bukti dari pasangan yang tertutup awan diberi bobot lebih kecil.
   - **Prior kelas.** Bias kelas Sinkhorn dengan target Earthquake 0,22. Kelas lain mengikuti porsi tingkat event, disusutkan 30% ke arah seragam. Estimasi tingkat event sendiri memberi Earthquake 0,21-0,26, konsisten dengan 0,22.

Porsi prediksi final: Volcano 949, Earthquake 702, Hurricane 536, Fire 533, Tsunami 272, Explosion 188.

Dicoba lalu ditolak (tercatat di notebook dan `code/`): kanonisasi hue berbasis histogram dan jaringan hue self-supervised (`08`, `09`), serta VLM tingkat event dengan mosaik 2x2 (`14`). Mosaik menurunkan kesepakatan dengan sinyal yang ditahan, dan 64% jawaban mentahnya Fire.

## Menjalankan inferensi tanpa melatih ulang

1. Pasang Python 3.13 dan PyTorch CUDA, lalu dependensi lain:

   ```powershell
   pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
   ```

   ```powershell
   pip install -r requirements.txt
   ```

2. Taruh data kompetisi di `Datasets/` dengan struktur asli: `Datasets/Fenomena/<Id>/{pre,post}.jpg` dan `Datasets/sample_submission.csv`.
3. Buka notebook dan jalankan semua sel. Bobot model pralatih diunduh otomatis ke `.cache/hf`: `facebook/dinov2-with-registers-large`, `timm/vit_large_patch14_clip_336.openai`, `Qwen/Qwen3-VL-8B-Instruct` (sekitar 17 GB) dan `Qwen/Qwen3-VL-4B-Instruct` (sekitar 9 GB).

Setiap langkah berat menyimpan hasilnya di `exp/cache/` dan dilewati bila cache sudah ada. Waktu di RTX 3060 12 GB:

| Langkah | Waktu |
|---|---|
| Survei noise dan cache citra bersih (CPU, 12 proses) | sekitar 7 menit |
| Embedding DINOv2 dan CLIP | sekitar 35 menit |
| VLM 8B (warna) | 1 jam 57 menit |
| VLM 4B (grayscale) | 1 jam 13 menit |
| Label model dan submission | di bawah 1 menit |

Dengan cache lengkap, seluruh notebook selesai dalam sekitar 30 detik. Butuh GPU minimal 12 GB dan ruang disk sekitar 45 GB (cache citra 10 GB, bobot model sekitar 30 GB, paket Python).

Untuk jalur cepat tanpa VLM, sertakan berkas fitur kecil dari `exp/cache/` (sekitar 475 MB): `emb_*.npz`, `clipzs_*.npz`, `vlm_q8b_clean512.npz`, `vlm_q4b_gray.npz`, `noise_survey.csv`, `cloud64*.npy`, `valid64*.npy`, `colour_covariates.npz`, `hdb_labels.npy`. Cache citra 512 piksel akan dibangun ulang dalam sekitar 5 menit. Hasil VLM bisa sedikit berbeda di GPU lain karena aritmetika bf16 dan 4-bit.

## Keterbatasan

- Akurasi zero-shot per pasangan terbatas dan tidak bisa diukur tanpa label.
- Pasangan kelas yang paling rawan tertukar: Fire vs Volcano di lereng gersang, Earthquake vs Explosion di kota, dan Hurricane vs Tsunami di pesisir.
- Porsi kelas selain Earthquake tidak diketahui dan diperkirakan dari data. Porsi Volcano (30%) dan Explosion (6%) adalah bagian yang paling tidak pasti.
