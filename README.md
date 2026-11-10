# Hauling KM-18

Source pipeline dan notebook pengujian hauling KM-18.

## Isi

- `hauling/`: deteksi kendaraan, tracking, pencarian panel nomor lambung, OCR, pengukuran kecepatan, overspeed, dan keluaran event.
- `tools/rnd_full_pipeline.py`: runner lima konfigurasi pengujian.
- `tests/`: pemeriksaan integrasi adapter dan runner yang tersedia pada snapshot.
- `notebooks/RnD_Full_Pipeline_Tracking_T4.ipynb`: notebook benchmark Google Colab.
- `code_manifest.json`: SHA256 file source dari snapshot notebook.

Source disalin dari snapshot notebook benchmark, bukan dari salinan yang ditambahkan instrumentasi diagnosis. Semua 31 hash source diverifikasi saat paket dibuat. Verifikasi hash bukan pengujian ulang inferensi.

## Menjalankan pada Google Colab

1. Buka notebook dalam folder `notebooks` di Google Colab.
2. Pilih GPU T4 dan jalankan instalasi dependency serta pemeriksaan lingkungan. Restart jika diminta.
3. Jalankan ekstraksi snapshot kode.
4. Upload paket aset pendamping `Aset_RnD_Full_Pipeline_Colab.zip` pada sel aset.
5. Jalankan pengujian sesuai petunjuk notebook.

Untuk pengujian hanya konfigurasi yang dipilih, ubah daftar varian menjadi:

```python
VARIANTS = ['pytorch_fp16']
```

Varian `pytorch_fp16` menggunakan ByteTrack dengan detector kendaraan PyTorch FP16. Model pencarian panel, OCR, dan classifier kendaraan tetap FP32. Notebook aslinya juga berisi tahap build TensorRT; tahap tersebut hanya diperlukan jika varian TensorRT akan diuji.

## Aset eksternal

Bobot model, video, konfigurasi kamera, dan referensi panel tidak disertakan dalam paket source. Gunakan paket aset yang sesuai dengan hash pada notebook, melalui akses yang disepakati dengan perusahaan. Paket aset berisi folder `model-requirement`, `reference`, `video`, dan `assets_manifest.json`.

## Hasil pengujian

Pada sampel 45 detik yang ditinjau, ByteTrack + PyTorch FP16 mencatat enam teks nomor lambung berstatus lengkap. Kelengkapan format tidak otomatis membuktikan identitas benar. Hasil tidak menjamin kinerja yang sama pada video lain.

## Batas snapshot

Notebook menyimpan snapshot source di dalam sel SOURCE_B64. Source `.py` dalam repository disediakan untuk review dan pengembangan. Jika source `.py` diedit, snapshot tertanam di notebook tidak otomatis berubah. Sinkronkan snapshot atau sesuaikan alur Colab agar membaca checkout repository ketika menguji perubahan tersebut.
