# IrisLens AI - GUI Modul Ekspansi Slit Lamp Berbasis AI

Aplikasi Desktop GUI Medis Profesional berbasis **Python** & **PyQt5** untuk skrining katarak dengan modul ekspansi slit lamp cerdas.

---

## 📸 Tinjauan Desain Layar

Aplikasi ini mengimplementasikan tata letak 3 halaman sesuai spesifikasi medis resolusi tinggi (1920x1080 / 16:9):

### 1. Halaman 1: Splash Loading Screen
- **Background**: Light clinical background (`#F4F7F6`).
- **Logo**: Simbol minimalis mata & lensa slit-lamp berwarna Deep Navy (`#0B2545`) dikelilingi oleh animasi *circular loading ring* berputar halus (60 FPS) berwarna Vibrant Cyan (`#00D2C8`).
- **Judul**: `IrisLens` (Deep Navy `#0B2545`) dan `AI` (Vibrant Cyan `#00D2C8`).
- **Subjudul**: *"Smart Optics for Precision Cataract Screening"*.
- **Footer**: *"Modul Ekspansi Slit Lamp Berbasis AI untuk Skrining Katarak"*.
- **Interaksi**: Auto-transisi setelah 3.5 detik, atau klik di mana saja untuk melewati.

### 2. Halaman 2: Registrasi Pasien Baru
- **Header**: Bilah horizontal Deep Navy (`#0B2545`) dengan teks *"IrisLens AI"* di kiri dan *"Registrasi Pasien Baru"* di tengah.
- **Form Card**: Wadah form 2x2 grid dengan kartu abu-abu lembut bertepi membulat:
  1. `ID PASIEN` (diisi manual)
  2. `NAMA PASIEN`
  3. `UMUR`
  4. `JENIS KELAMIN` (Pilihan *Laki-laki* / *Perempuan*)
- **Tombol Aksi**:
  - `BATAL / RESET` (Secondary Grey `#6D7C8A`)
  - `LANJUT KE PEMERIKSAAN` (Vibrant Cyan `#00D2C8`) hanya melanjutkan jika ID, nama, umur, dan jenis kelamin sudah diisi.

### 3. Halaman 3: Medical Examination Dashboard
- **Header**: Deep Navy `#0B2545` dengan teks *"IrisLens AI"*, tombol kembali ke registrasi, serta status chip ringkasan data pasien saat ini.
- **Panel Kiri (Video Feed & Kontrol)**:
  - Preview video langsung dari kamera yang dipilih.
  - Badge sudut kiri atas menunjukkan status model dan kelas yang terdeteksi.
  - 3 Tombol Kontrol:
    - `MULAI` (Cyan `#00D2C8`) — Menjalankan / melanjutkan stream video.
    - `STOP` (Putih berbingkai Navy) — Menjeda / freeze frame video.
    - `CAPTURE` (Deep Navy `#0B2545`) — Mengambil snapshot bersih dan menjalankan inferensi YOLO26. Pratinjau capture hanya menampilkan gambar asli; kelas dan confidence ditampilkan di panel hasil.
  - Pemilih sumber menampilkan webcam/USB slit-lamp yang terdeteksi melalui OpenCV; jika kamera tidak tersedia, layar menampilkan pemberitahuan.
- **Panel Kanan (Hasil Deteksi AI)**:
  - Judul: `HASIL DETEKSI AI`
  - Hasil kelas, confidence, dan jumlah objek berasal dari prediksi model.
  - Data ID, nama, umur, dan jenis kelamin dari formulir tampil di header halaman pemeriksaan.
  - Tombol `KIRIM KE DOKTER` menampilkan ringkasan; pengiriman EMR belum terhubung.

---

## 🛠️ Persyaratan Sistem

- Python 3.9+ (sudah teruji pada Python 3.14)
- Dependensi:
  ```bash
  pip install -r requirements.txt
  ```

---

## 🚀 Cara Menjalankan Aplikasi

Jalankan perintah berikut di terminal / PowerShell:

```bash
python main.py
```

### Menjalankan UI dengan model hasil training

UI menjalankan inferensi YOLO26 langsung dengan runtime NCNN, tanpa memuat PyTorch. Instal dependensi GUI dari root proyek:

```bash
python3 -m pip install --user --break-system-packages -r GUI/GUI/requirements.txt
```

Checkpoint training saat ini masih berupa `best.pt`, jadi ekspor sekali ke NCNN dari root proyek menggunakan environment training yang memiliki Ultralytics/PyTorch:

```bash
python3 train/export_ncnn.py
```

Ekspor menghasilkan folder `train/runs/cataract-yolo26n/weights/best_ncnn_model/`. Proses konversi `.pt` ke NCNN memakai PyTorch sebagai alat ekspor; aplikasi GUI setelah ekspor hanya memakai NCNN. Jalankan UI dari root proyek:

```bash
python3 GUI/GUI/main_yolo.py
```

Secara default UI memuat folder NCNN di atas. Untuk memakai model NCNN lain, atur variabel lingkungan `IRISLENS_NCNN_MODEL` ke folder yang berisi `model.ncnn.param` dan `model.ncnn.bin`. Pilih sumber kamera, lalu tekan **CAPTURE**. Pratinjau menampilkan gambar asli tanpa kotak atau label; kelas, confidence, dan jumlah objek tampil di panel hasil. Kelas model adalah `immature`, `mature`, dan `normal`. Data pasien dari halaman registrasi diteruskan ke header pemeriksaan.

---

## 📂 Struktur File

```
GUI/
├── assets/
│   ├── eye_feed_sample.jpg      # Gambar macro slit-lamp resolusi tinggi
│   ├── eye_symbol.png           # Simbol vektor mata & celah slit lamp
│   ├── logo_full.png            # Master logo
│   └── logo_transparent.png     # Logo transparan
├── main.py                      # Skrip utama aplikasi GUI PyQt5 lengkap
├── render_screens.py            # Skrip utilitas untuk merender screenshot 3 layar
├── requirements.txt             # Daftar dependensi library
└── README.md                    # Dokumentasi lengkap
```

