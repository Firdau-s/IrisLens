# Training YOLO26 untuk deteksi katarak

Program ini melatih model deteksi YOLO26 pada dataset lokal `cataract detection.v3i.yolo26`. Dataset memiliki tiga kelas: `immature`, `mature`, dan `normal`.

## Persiapan

Jalankan perintah dari folder proyek:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r train/requirements.txt
```

## Mulai training

```bash
python train/train.py
```

Default memakai bobot pretrained `yolo26n.pt`, 100 epoch, ukuran gambar 640, dan batch 16. Bobot pretrained akan diunduh otomatis oleh Ultralytics saat pertama kali dijalankan, sehingga koneksi internet dibutuhkan.

Contoh pengaturan:

```bash
python train/train.py --model yolo26s.pt --epochs 150 --imgsz 640 --batch 8 --device 0
python train/train.py --device cpu
```

Pilihan lengkap tersedia melalui:

```bash
python train/train.py --help
```

Hasil training tersimpan di `train/runs/<task>/<nama-run>/`, termasuk bobot `weights/best.pt` dan `weights/last.pt`. Skrip memakai split train dan valid dari `data.yaml`; split test dataset tidak dipakai selama training.

## Ekspor ke NCNN

Untuk menjalankan hasil training melalui runtime NCNN, ekspor checkpoint sekali setelah training selesai:

```bash
python train/export_ncnn.py
```

File sumber training tetap berupa `.pt`; proses konversi memakai Ultralytics/PyTorch. Aplikasi GUI setelah ekspor memuat `model.ncnn.param` dan `model.ncnn.bin` langsung menggunakan NCNN, tanpa menjalankan inferensi PyTorch.
