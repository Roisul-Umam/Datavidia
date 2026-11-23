# 📊 Datavidia - Arkavidia Data Science Competition

Repository ini berisi kumpulan kode, eksperimen, *exploratory data analysis* (EDA), dan pemodelan *machine learning* untuk kompetisi **Datavidia** (bagian dari rangkaian acara Arkavidia ITB).

---

## 👥 Tim: bikin ai pake ai

* **Farel Boston** - Data Preprocessing & Feature Engineering
* **Rheza Abdilla** - Modeling & Hyperparameter Tuning
* **Roisul Umam** - EDA & Ensemble

---

## 📁 Struktur Repository

```text
Datavidia/
├── warmup/                 # Kode & eksperimen tahap pemanasan (Warmup)
│   ├── eda.ipynb           # Eksplorasi data awal
│   └── baseline.ipynb      # Model baseline
├── preliminary/            # Kode & eksperimen tahap penyisihan (opsional)
├── data/                   # Folder dataset (diabaikan di .gitignore jika ukuran besar)
├── submissions/            # Hasil prediksi (.csv) yang di-submit ke platform kompetisi
├── requirements.txt        # Daftar library Python yang digunakan
└── README.md               # Dokumentasi repository
```

---

## 🎯 Deskripsi Masalah & Pendekatan

* **Tujuan:** [Jelaskan singkat masalah yang diselesaikan, misal: Memprediksi tingkat churn pelanggan / klasifikasi gambar / forecasting penjualan].
* **Metrik Evaluasi:** [Misal: RMSE / F1-Score / ROC-AUC / Accuracy].
* **Pendekatan Utama:**
  1. **Data Cleaning & Preprocessing:** Penanganan *missing values*, *outlier*, dan encoding fitur kategorikal.
  2. **Feature Engineering:** Pembuatan fitur baru berbasis domain knowledge dan agregasi.
  3. **Modeling:** Eksperimen menggunakan [misal: LightGBM, XGBoost, CatBoost, Neural Network].
  4. **Validation Strategy:** Menggunakan [misal: Stratified K-Fold / TimeSeriesSplit] untuk mencegah *data leakage*.

---

## 🛠️ Instalasi & Cara Menjalankan

1. **Clone repository ini:**
   ```bash
   git clone https://github.com[username-github-kamu]/[nama-repo-kamu].git
   cd Datavidia
   ```

2. **Buat virtual environment & install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Untuk Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Jalankan Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```
   Buka file notebook di dalam folder `warmup/` atau folder tahap kompetisi terkait.

---

## 📈 Ringkasan Hasil Eksperimen (Log)

| Versi | Model / Pendekatan | Validation Score | Public LB Score | Catatan |
| :--- | :--- | :---: | :---: | :--- |
| v1 | Baseline (Random Forest) | `0.xxxx` | `0.xxxx` | Tanpa feature engineering |
| v2 | LightGBM + Feature Eng. | `0.xxxx` | `0.xxxx` | Penambahan fitur agregasi |
| v3 | Ensemble (LGBM + CatBoost) | `0.xxxx` | `0.xxxx` | Blending prediksi terbaik |

---

## 📝 Catatan
* Dataset kompetisi tidak disertakan secara langsung di repository ini sesuai dengan aturan kompetisi / batas ukuran file GitHub.
