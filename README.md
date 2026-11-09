# IrisLens AI

**IrisLens AI** adalah proyek prototipe modul ekspansi slit lamp berbasis kecerdasan buatan untuk membantu skrining katarak. Aplikasi menggunakan model deteksi YOLO26 untuk menganalisis citra mata dan menampilkan kelas prediksi beserta tingkat kepercayaannya.

> **Catatan:** IrisLens AI adalah proyek prototipe dan alat bantu skrining. Hasil prediksi tidak menggantikan pemeriksaan maupun diagnosis oleh tenaga kesehatan profesional.

## Fitur

- Antarmuka desktop berbasis Python dan PyQt5.
- Formulir data pasien sebelum pemeriksaan.
- Pemilihan kamera, termasuk webcam atau kamera slit lamp yang dikenali OpenCV.
- Pengambilan gambar (*capture*) dan inferensi model YOLO26.
- Tampilan kelas prediksi, confidence, dan jumlah objek yang terdeteksi.
- Runtime inferensi NCNN agar aplikasi desktop dapat menjalankan model tanpa memuat PyTorch.
- Skrip untuk melatih model dan mengekspor checkpoint ke format NCNN.

Kelas model yang digunakan: `immature`, `mature`, dan `normal`.

## Sumber dataset

Dataset deteksi katarak diperoleh dari Roboflow Universe:

[Cataract Detection — Roboflow Universe](https://universe.roboflow.com/crowdmanagement-6h2sr/cataract-detection-s9s5t)

Dataset disusun dalam format YOLO dengan pembagian `train`, `valid`, dan `test`. Periksa halaman Roboflow untuk detail atribusi, lisensi, dan versi dataset. Untuk menjalankan skrip training, letakkan hasil unduhan di folder `cataract detection.v3i.yolo26/` pada direktori utama proyek.

## Struktur proyek

```text
.
├── GUI/GUI/                 # Aplikasi desktop, aset, dan runtime NCNN
├── train/                   # Skrip training dan ekspor model
├── web_app.py               # Server backend untuk pencatatan pemeriksaan
├── jalankan_irislens.sh     # Skrip peluncur desktop
└── README.md
```

## Persiapan

Gunakan Python 3.9 atau lebih baru. Perintah berikut dijalankan dari direktori utama proyek.

Pasang dependensi aplikasi desktop:

```bash
python3 -m pip install -r GUI/GUI/requirements.txt
```

Jalankan aplikasi desktop:

```bash
python3 GUI/GUI/main_yolo.py
```

Jika model NCNN berada di lokasi berbeda, atur `IRISLENS_NCNN_MODEL` ke folder yang berisi `model.ncnn.param` dan `model.ncnn.bin`:

```bash
export IRISLENS_NCNN_MODEL="/path/ke/folder/model-ncnn"
python3 GUI/GUI/main_yolo.py
```

## Training dan ekspor model

Pasang dependensi training:

```bash
python3 -m pip install -r train/requirements.txt
```

Mulai training dengan konfigurasi bawaan:

```bash
python3 train/train.py
```

Bobot dasar YOLO26 akan diunduh otomatis oleh Ultralytics bila belum tersedia. Hasil training disimpan di `train/runs/`. Untuk mengekspor checkpoint hasil training ke NCNN:

```bash
python3 train/export_ncnn.py
```

Opsi training seperti jumlah epoch, ukuran gambar, batch, dan perangkat dapat dilihat dengan `python3 train/train.py --help`. Panduan lebih lengkap tersedia di [`train/README.md`](train/README.md).

## Backend web

`web_app.py` menyediakan backend lokal untuk menyimpan data pemeriksaan dan gambar capture. Kredensial login wajib diberikan melalui environment variable dan tidak disimpan di source code.

```bash
export IRISLENS_USERNAME="nama-pengguna"
export IRISLENS_PASSWORD="kata-sandi-kuat"
python3 web_app.py
```

Server menggunakan port `8000` secara default. Port dapat diubah melalui `IRISLENS_PORT`; lokasi penyimpanan data dapat diatur melalui `IRISLENS_DATA_DIR`.

## Lisensi dan atribusi

Ikuti lisensi serta ketentuan atribusi yang tercantum pada halaman sumber dataset Roboflow. Pastikan penggunaan dan distribusi ulang data mematuhi ketentuan pemilik dataset.
