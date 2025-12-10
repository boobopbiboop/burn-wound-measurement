# Sistem Deteksi & Pengukuran Luka Bakar

Aplikasi computer vision untuk deteksi otomatis, segmentasi, dan pengukuran luka bakar menggunakan teknik pemrosesan gambar.

## Anggota Tim

| Nama                  | NRP        |
|-----------------------|------------|
| Athaya Rohadatul Yaqutah | 5025221235 |
| Agnes                 | 502522     |
| Nadya Saraswati Putri | 5025221246 |

## Gambaran Proyek

Proyek ini mengimplementasikan sistem otomatis untuk analisis luka bakar menggunakan teknik computer vision. Sistem ini menyediakan:

- **Preprocessing Gambar**: Meningkatkan kualitas gambar luka untuk analisis yang lebih baik
- **Segmentasi Luka**: Secara otomatis mengidentifikasi dan mensegmentasi area luka bakar
- **Kalkulasi Pengukuran**: Menghitung dimensi dan area luka
- **Interface Web Interaktif**: Aplikasi Streamlit yang user-friendly untuk analisis real-time

## Fitur

- 🔥 Deteksi luka bakar otomatis
- 📏 Kalkulasi pengukuran luka yang presisi
- 🖼️ Preprocessing dan peningkatan kualitas gambar
- 🎯 Algoritma segmentasi canggih
- 📊 Visualisasi dan ekspor hasil
- 🌐 Interface berbasis web untuk akses mudah

## Struktur Proyek

```
burn-wound-measurement/
├── src/                    # Source code
│   └── app.py             # Aplikasi Streamlit utama
├── notebooks/             # Jupyter notebooks
│   ├── step_1_preprocessing.ipynb
│   ├── step_2_segmented.ipynb
│   └── step_3_measurement.ipynb
├── data/                  # Direktori data
│   └── processed/         # Dataset yang telah diproses
│       ├── augmented/     # Gambar yang telah diaugmentasi
│       ├── measured/      # Hasil pengukuran
│       ├── segmented/     # Gambar tersegmentasi
│       └── SELECTED_200/  # Dataset terpilih
├── docs/                  # Dokumentasi
├── assets/                # Aset proyek
├── requirements.txt       # Dependencies Python
└── README.md             # Dokumentasi proyek
```

## Instalasi

1. Clone repository:
```bash
git clone <repository-url>
cd burn-wound-measurement
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Penggunaan

### Menjalankan Aplikasi Web
```bash
streamlit run src/app.py
```

### Menggunakan Jupyter Notebooks
1. **Preprocessing**: `notebooks/step_1_preprocessing.ipynb`
2. **Segmentasi**: `notebooks/step_2_segmented.ipynb`
3. **Pengukuran**: `notebooks/step_3_measurement.ipynb`

## Stack Teknologi

- **Python 3.x**
- **OpenCV**: Pemrosesan gambar dan computer vision
- **Streamlit**: Framework aplikasi web
- **NumPy**: Komputasi numerik
- **Pandas**: Manipulasi data
- **PIL/Pillow**: Penanganan gambar
- **Google APIs**: Integrasi cloud

## Metodologi

1. **Preprocessing Gambar**: Pengurangan noise, peningkatan kontras, dan normalisasi
2. **Segmentasi Luka**: Algoritma canggih untuk mengisolasi area luka bakar
3. **Ekstraksi Fitur**: Mengekstrak karakteristik luka yang relevan
4. **Kalkulasi Pengukuran**: Menghitung area, perimeter, dan metrik dimensi
5. **Visualisasi**: Menghasilkan hasil beranotasi dan laporan

## Memulai

1. Install dependencies yang diperlukan
2. Jalankan aplikasi Streamlit: `streamlit run src/app.py`
3. Upload gambar luka bakar melalui interface web
4. Lihat hasil analisis otomatis dan pengukuran

## Kontribusi

Proyek ini merupakan bagian dari tugas mata kuliah Computer Vision. Untuk pertanyaan atau kontribusi, silakan hubungi anggota tim yang tercantum di atas.
