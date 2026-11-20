# Three-Class Forensic Classification of Anime Images (Real, Inpainting, Text2Image)

Repositori ini berisi implementasi kode pelatihan, evaluasi, audit dataset, dan aplikasi web interaktif untuk klasifikasi forensik tiga kelas pada ilustrasi bergaya anime: **Real** (buatan manusia), **Inpainting** (manipulasi sebagian oleh AI), dan **Text2Image** (buatan AI sepenuhnya).

Proyek ini membandingkan kinerja klasifikasi dan efisiensi komputasi dari arsitektur **CNN**, **Vision Transformer**, dan **Hybrid** di bawah satu protokol eksperimen yang terkontrol.

---

## 📋 Daftar Isi
- [Latar Belakang](#latar-belakang)
- [Hasil Utama Eksperimen](#hasil-utama-eksperimen)
- [Struktur Direktori](#struktur-direktori)
- [Instalasi & Persyaratan](#instalasi--persyaratan)
- [Persiapan Dataset](#persiapan-dataset)
- [Cara Menjalankan Proyek](#cara-menjalankan-proyek)
  - [1. Audit Shortcut Metadata](#1-audit-shortcut-metadata)
  - [2. Pelatihan Model (Training)](#2-pelatihan-model-training)
  - [3. Evaluasi & Profiling Perangkat Keras](#3-evaluasi--profiling-perangkat-keras)
  - [4. Aplikasi Web Detektor (FastAPI)](#4-aplikasi-web-detektor-fastapi)
- [Sitasi & Lisensi](#sitasi--lisensi)

---

## 🔍 Latar Belakang
Sebagian besar detektor gambar AI memperlakukan keaslian gambar sebagai masalah biner (*Real* vs *AI*) dan sering melewatkan manipulasi lokal melalui teknik *inpainting*. Proyek ini mengevaluasi seberapa baik berbagai paradigma arsitektur visi komputer membedakan ketiga kelas asal-usul gambar pada dataset **AniPrO** (15.000 gambar seimbang):
1. **Real**: Ilustrasi yang digambar sepenuhnya oleh manusia.
2. **Inpainting**: Gambar manusia yang sebagian areanya diregenerasi oleh model difusi.
3. **Text2Image**: Gambar yang dihasilkan sepenuhnya dari *prompt* teks (FLUX.1, Stable Diffusion, SDXL).

---

## 📊 Hasil Utama Eksperimen
Evaluasi dilakukan pada **1.500 gambar uji (*in-domain*)** menggunakan ukuran input `224×224` pada GPU NVIDIA GeForce RTX 3050 Laptop (6 GB) dengan *batch size* 32 dan *mixed precision*:

| Model | Paradigma | Parameter | GFLOPs | Akurasi 3-Kelas | Akurasi Biner | Throughput | Peak VRAM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Swin-Tiny** | Hierarchical ViT | 27,52 Juta | 4,508 | **97,67%** | **98,13%** | 21,12 img/s | 618,68 MB |
| **MaxViT-Tiny** | Hybrid (MBConv + Multi-Axis) | 30,41 Juta | 5,421 | 97,40% | 97,87% | 34,78 img/s | 913,25 MB |
| **CvT-21** | Hybrid (Conv. Token + Attn) | 31,24 Juta | 7,206 | 97,20% | 97,47% | **35,64 img/s** | 946,79 MB |
| **ConvNeXt-Tiny** | Modernized CNN | 27,82 Juta | 4,470 | 95,27% | 95,73% | 31,79 img/s | **499,18 MB** |

> **Catatan:** Sebagian besar kesalahan klasifikasi pada seluruh model terjadi antara kelas **Real** dan **Inpainting**, sedangkan kelas **Text2Image** terdeteksi hampir sempurna (recall 99,2% – 99,8%).

---

## 📁 Struktur Direktori

```text
├── dataset/
│   └── splits/
│       ├── train/          # ID 1 - 4.000 (12.000 gambar)
│       ├── validation/     # ID 4.001 - 4.500 (1.500 gambar)
│       └── test/           # ID 4.501 - 5.000 (1.500 gambar)
├── models/                 # Menyimpan bobot model hasil training (.pth)
├── results/                # Menyimpan log histori epoch & ringkasan evaluasi (.csv)
└── src/
    ├── audit_metadata.py   # Audit shortcut metadata menggunakan Decision Tree
    ├── train.py            # Skrip pelatihan mandiri untuk seluruh arsitektur
    ├── evaluate.py         # Evaluasi metrik klasifikasi & profiling RAM/VRAM/FLOPs
    └── app.py              # Aplikasi web antarmuka pengguna berbasis FastAPI
```

---

## ⚙️ Instalasi & Persyaratan

Pastikan Anda menggunakan Python 3.9+ dan PyTorch dengan dukungan CUDA (disarankan). Instal pustaka yang diperlukan dengan perintah berikut:

```bash
pip install torch torchvision timm transformers fastapi uvicorn python-multipart pillow scikit-learn pandas numpy psutil fvcore
```

---

## 🗂️ Persiapan Dataset

1. Unduh dataset **AniPrO** dari repositori resmi Hugging Face: https://huggingface.co/datasets/YAN-LIU05/AniPro (Lisensi CC BY-NC 4.0).
2. Bagi dataset dengan rasio **80:10:10 berdasarkan indeks ID** untuk mencegah kebocoran data (*data leakage*) antar gambar berpasangan:
   - `train`: ID `1` hingga `4000`
   - `validation`: ID `4001` hingga `4500`
   - `test`: ID `4501` hingga `5000`
3. Susun subfolder di dalam masing-masing split menjadi: `real/`, `inpainting/`, dan `text2image/`.

---

## 🚀 Cara Menjalankan Proyek

*(Sesuaikan nama file skrip di bawah ini dengan nama file aktual di dalam folder `src/` Anda).*

### 1. Audit Shortcut Metadata
Memeriksa bias struktural (dimensi resolusi, rasio aspek, ukuran file) sebelum pelatihan model visi:
```bash
python src/audit_metadata.py
```

### 2. Pelatihan Model (Training)
Melatih arsitektur (Swin-Tiny, ConvNeXt-Tiny, MaxViT-Tiny, CvT-21, serta dukungan EfficientNetV2-S & CoAtNet) dengan *optimizer* AdamW, *cosine annealing*, dan augmentasi kompresi JPEG acak:
```bash
python src/train.py
```
*Bobot terbaik otomatis disimpan ke folder `models/` dan log pelatihan dicatat ke `results/`.*

### 3. Evaluasi & Profiling Perangkat Keras
Menghitung akurasi 3-kelas, akurasi biner, *confusion matrix*, GFLOPs, latensi, *throughput*, serta penggunaan puncak VRAM dan RAM:
```bash
python src/evaluate.py
```
*Hasil evaluasi disimpan ke `results/evaluation_summary_model_baru.csv` dan `results/efficiency_summary_model_baru.csv`.*

### 4. Aplikasi Web Detektor (FastAPI)
Menjalankan antarmuka web lokal untuk menguji deteksi gambar menggunakan model terbaik (**Swin-Tiny**):
```bash
python src/app.py
```
Buka peramban (*browser*) dan akses **`http://127.0.0.1:8000`**. Unggah gambar ilustrasi untuk melihat probabilitas masing-masing kelas (`Real`, `Inpainting`, `Text2Image`) beserta keputusan biner.

---

## 📚 Sitasi & Lisensi

Jika Anda menggunakan kode atau mengacu pada eksperimen ini, silakan sitasi karya berikut:

```bibtex
@article{putra2026threeclass,
  title={Three-Class Forensic Classification of Anime Images (Real, Inpainting, Text2Image) Using CNN, Vision Transformer, and Hybrid Architectures},
  author={Putra, Ricky Eka and Aziz, Muhammad Naufal},
  institution={Department of Informatics, Universitas Negeri Surabaya},
  year={2026}
}
```

- **Kode Sumber**: Dirilis untuk tujuan penelitian dan akademik.
- **Dataset AniPrO**: Tunduk pada lisensi **CC BY-NC 4.0**.
