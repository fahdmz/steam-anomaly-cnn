# Context Proyek: Visualisasi Time Series dengan CNN

## Tentang tugas

Tugas kelompok (2 orang) mata kuliah dari Pak Endra Joelianto, Institut Teknologi Bandung. Instruksinya: membuat visualisasi data time series menggunakan CNN. Data bebas diambil dari mana saja. Keluaran yang dinilai adalah visualisasi dan insight yang didapatkan dari data yang divisualisasikan.

Tingkat ketelitian yang diinginkan: setara tugas kuliah S1, bukan riset lanjut. Pendekatan sederhana, jelas, dan mudah dijelaskan ulang oleh mahasiswa. Hindari analisis atau struktur kode yang berlebihan.

Keluaran proyek ini ada dua:
1. Kode: satu atau dua notebook Jupyter yang dijalankan dari atas ke bawah.
2. Paper LaTeX berbahasa Indonesia yang melaporkan metode, visualisasi, hasil, dan insight.

## Paper acuan utama

Rahadian, H., Bandong, S., Widyotriatmo, A., Joelianto, E. (2023). Image encoding selection based on Pearson correlation coefficient for time series anomaly detection. Alexandria Engineering Journal, 82, 304-322. https://doi.org/10.1016/j.aej.2023.09.070

File PDF ada di `refs/`. Pak Endra adalah salah satu penulisnya, jadi metode proyek ini mengikuti paper tersebut.

Ringkasan metode paper:
1. Data: tegangan motor DC (INA219, Raspberry Pi, sampling 200 ms), 45.000 titik. Anomali dibuat dengan menyisipkan nilai over-voltage acak (5,001 sampai 5,999 V), 1 sampai 10 titik per segmen, posisi acak (point anomaly).
2. Segmentasi: segmen 20 titik, 2.250 pasangan segmen normal dan anomali.
3. Image encoding dengan `pyts`: GADF, GASF, dan MTF. MTF divariasikan berdasarkan `strategy` (uniform, quantile, normal) dan `n_bins` (2, 4, 5, 10), total 14 variasi. Parameter lain default (`image_size=1.0`, `sample_range=(-1, 1)`, `overlapping=False`, `flatten=False`).
4. Matriks diubah menjadi gambar dengan colormap `rainbow` (origin lower), lalu di-resize ke 150 x 150.
5. CNN: 3 blok Conv2D (16, 32, 64 filter, kernel 3x3, ReLU) dengan pooling, Dense 512 ReLU, output 1 unit sigmoid. RMSprop lr 0.001, binary cross-entropy, batch 20, 20 epoch.
6. Analisis: hitung mean, SD, dan jumlah nilai unik (count) dari matriks hasil encoding untuk data normal dan anomali, lalu Δmean, ΔSD, Δcount (selisih absolut rata-rata). Korelasi Pearson antara tiap Δ dan akurasi test CNN (14 titik, satu per encoding).
7. Hasil paper: ΔSD paling berpengaruh (r = 0,944), lalu Δmean (r = 0,705), lalu Δcount (r = 0,455). GADF, GASF, dan MTF Uniform mencapai akurasi di atas 90%, MTF Quantile sekitar 66 sampai 73%, MTF Normal 50% karena semua gambarnya identik (merah penuh). Akurasi di atas 90% didapat ketika ΔSD lebih dari sekitar 0,125. Melewati konversi gambar dan training CNN menghemat waktu komputasi sekitar 95%.
8. Validasi dengan 3 dataset baru. Urutan pengaruh (ΔSD > Δmean > Δcount) tetap konsisten.

Paper kedua di `refs/` (Joelianto & Sagala 2012, Swarm Tracking Control for Flocking of a Multi-Agent System) hanya konteks tambahan dari dosen yang sama dan tidak dipakai sebagai metode.

## Kontribusi proyek ini

Menerapkan pipeline paper pada data jumlah pemain game Steam, lalu menambahkan sisi visualisasi:
1. Cek apakah temuan paper (ΔSD paling berpengaruh) juga berlaku pada data ini. Hasil yang berbeda dilaporkan apa adanya.
2. Perbandingan visual pasangan gambar normal dan anomali per encoding, seperti Fig. 7 dan Fig. 8 di paper.
3. Grad-CAM untuk 2 sampai 3 contoh segmen anomali, dipetakan balik ke hari pada plot segmen asli.
4. Insight dari data: pola mingguan, lonjakan asli (update besar, Steam Sale, rilis) dibahas di bagian Insight, tidak dijadikan label.

Tidak dikerjakan (sengaja dipangkas): Spearman, p-value, skor ketepatan Grad-CAM, t-SNE/UMAP, silhouette score, ablation panjang segmen, studi kasus dataset kedua.

## Keputusan (final, ditetapkan user)

1. Dataset: jumlah pemain bersamaan harian game Steam dari repo GitHub `ibrhr/Steam-Game-Analytics`, folder `data/`, file `player_counts.csv` per game (sumber asli SteamDB). Enam game: Dark Souls 3, Euro Truck Simulator 2, Garry's Mod, Rainbow Six Siege, Stardew Valley, Terraria. File `prices.csv` tidak dipakai.
2. Panjang segmen: 20 titik (20 hari). Tanpa ablation.
3. Paper: Bahasa Indonesia, IEEEtran conference, sekitar 4 sampai 6 halaman, bahasa lugas dan tidak terlalu teknis.
4. Framework: PyTorch di `.venv` proyek. Colormap MTF dengan rentang [0, 1] (GAF tetap [-1, 1]).
5. Tetap 14 encoding (GADF, GASF, MTF Uniform/Quantile/Normal dengan n_bins 2, 4, 5, 10).
6. Anomali default: disuntikkan seperti paper (lonjakan atau penurunan tajam pada 1 sampai 5 titik acak per segmen), setelah data dinormalisasi per game. Detail aturan suntikan menunggu persetujuan user setelah pengecekan data.

## Struktur folder

```
refs/            PDF paper acuan
data/raw/        CSV asli per game (jangan diubah)
data/processed/  segmen dan label hasil olahan
notebooks/       1-2 notebook, dijalankan dari atas ke bawah
models/          bobot CNN terlatih per encoding (.pt, sekitar 36 MB per file)
figures/         figur PNG (dipakai juga oleh paper)
results/         tabel CSV hasil
paper/           main.tex, refs.bib, sections/
```

Tidak ada folder `src/` dan tidak ada `run_all`. Semua kode ada di notebook, dengan sel markdown singkat yang menjelaskan tiap langkah. Environment ada di `.venv/` (versi library di `requirements.txt`); jalankan notebook dengan `.venv/bin/jupyter`.

Status: Tahap 1 selesai (`notebooks/01_data_segmentasi.ipynb`, keluaran di `data/processed/segmen.npz` dan `segmen_meta.csv`). 1.229 pasang segmen, split 860/184/185 (70/15/15, seed 42, stratifikasi per game). Anomali: 1 sampai 5 hari per segmen, faktor 2 sampai 5 kali (lonjakan) atau seperdua sampai seperlima (penurunan), disuntik pada jumlah pemain asli sebelum log10 dan standarisasi per game (statistik dari segmen normal di train).

Tahap 2 selesai (`notebooks/02_encoding.ipynb`): matriks 14 encoding untuk 2.458 segmen (1.229 normal lalu 1.229 anomali, urutan sama dengan `segmen_meta.csv`) tersimpan di `data/processed/matriks_encoding.npz` (kunci seperti `MTF_Uniform_5`, plus `y`). Gambar tidak disimpan, dibuat ulang di Tahap 3 dengan rainbow, origin lower, resize tetangga terdekat ke 150x150 (GAF rentang [-1, 1], MTF [0, 1]). Temuan: MTF Normal tidak merah semua pada data terstandar (gambar seragam 8,5% sampai 66,8% pada segmen normal, tergantung n_bins), count GAF hampir selalu mendekati batas maksimum sehingga Δcount nyaris nol.

Tahap 3 selesai (`notebooks/03_training_cnn.ipynb`): CNN sesuai paper (3 blok Conv 16/32/64 + pooling, Dense 512, RMSprop lr 0,001 dengan alpha 0,9 dan eps 1e-7 seperti default Keras, batch 20, 20 epoch, seed 42, loss BCEWithLogits) dilatih per encoding, tanpa tuning. Split: 1.720 gambar train, 368 validasi, 370 test. Model epoch ke-20 yang dievaluasi (tanpa pemilihan checkpoint). Hasil: `results/akurasi_encoding.csv`, `riwayat_training.csv`, `prediksi_test.csv`, `mtf_normal_akurasi.csv`, `waktu_per_tahap.csv`; bobot di `models/`. Akurasi test: GADF 97,8%, GASF 97,3%, MTF Uniform 82 sampai 93%, MTF Normal 75 sampai 81%, MTF Quantile 57 sampai 71%. Training memakai GPU Apple (MPS) sehingga hasil tidak persis sama bila diulang. Satu run per encoding dengan 370 gambar test, selisih di bawah sekitar 4 poin tidak bisa dianggap bermakna.

Tahap 4 selesai (`notebooks/04_korelasi_gradcam.ipynb`): korelasi Pearson Δ vs akurasi test (14 titik): Δmean 0,496, ΔSD 0,507, Δcount 0,412 (paper 0,705; 0,944; 0,455) di `results/korelasi_pearson.csv`. Peringkat sama dengan paper tetapi ΔSD dan Δmean hampir seri dan ΔSD jauh lebih lemah. Grad-CAM pada model GADF untuk 3 segmen anomali di test (2 benar, 1 salah) di `figures/13_gradcam_contoh.png` dan `results/gradcam_contoh.csv`. Penjelasan penyebab ada di sel Pembahasan notebook (MTF Uniform kuat tanpa ΔSD besar, MTF Normal tidak lagi degenerate, Δmean GADF dan Quantile sama-sama 0, Δcount GAF hampir konstan).

Tahap 5 selesai: paper di `paper/` (main.tex, sections/*.tex, refs.bib), 6 halaman, 6 figur dan 3 tabel, tanpa placeholder. IEEEtran.cls dan IEEEtran.bst (V1.8b dari CTAN) disalin ke `paper/` karena BasicTeX di mesin ini tidak menyertakannya. Kompilasi dari dalam folder `paper/`: pdflatex main, bibtex main, pdflatex main, pdflatex main. Figur diambil dari `../figures/`.

## Aturan teknis

1. Python 3, `pyts`, `numpy`, `pandas`, `matplotlib`, `scikit-learn` (untuk split dan metrik), PyTorch. Catat versi library di `requirements.txt`.
2. Seed tetap untuk semua proses acak. Split train/val/test sama untuk semua encoding supaya perbandingannya adil.
3. Kebocoran data dicegah di kode: segmen normal dan versi anomalinya selalu berada di split yang sama (split per pasangan), validasi tidak diambil dari data test, dan statistik normalisasi dihitung dari data normal saja. Cukup dijelaskan singkat di paper.
4. Komentar kode seminimal mungkin.
5. Figur memakai gaya matplotlib biasa, tidak perlu dipoles berlebihan, tapi tetap ada judul, label sumbu, dan legend jika perlu. PNG dpi 150 atau lebih.
6. Catat waktu komputasi tiap tahap (encoding, konversi gambar, training) seperti Table 14 di paper.
7. Jangan membuat branch, commit, push, atau membuka PR. Git diurus user sendiri.

## Catatan data (hasil pengecekan, 2026-11-04)

Data asli dan figur EDA awal sudah disalin ke `data/raw/` dan `figures/`. Hal yang harus ditangani di notebook:
1. Resolusi campuran: harian sampai 2025-05-21, lalu interval 1 jam dan 10 menit sampai 2025-06-21. Diseragamkan ke harian dengan nilai maksimum per hari.
2. Hari terakhir (2025-06-21) belum lengkap (data sampai 08:20), dibuang.
3. Dark Souls 3, Rainbow Six Siege, dan Stardew Valley diawali nilai 1 sampai 3 sebelum rilis. Dipotong mulai hari pertama dengan pemain >= 1000.
4. Garry's Mod sebelum 2012 sangat bolong (sampai 82% hilang), dipakai mulai 2012-01-01.
5. Tanggal hilang diisi interpolasi linear hanya jika celahnya paling banyak 3 hari. Segmen yang masih mengandung nilai hilang dibuang.
6. Kolom `Average Players` kebanyakan kosong dan tidak dipakai.

## Aturan penulisan paper

1. LaTeX dengan IEEEtran (conference), dipecah per section di `paper/sections/`. Referensi di `refs.bib` dengan BibTeX.
2. Struktur: Abstrak, Pendahuluan, Tinjauan Pustaka singkat (GAF, MTF, CNN, korelasi Pearson), Metodologi, Hasil dan Pembahasan, Insight, Kesimpulan, Referensi. Total sekitar 4 sampai 6 halaman.
3. Gaya tulisan narasi akademik dalam paragraf yang lugas, bukan daftar bullet dan tidak terlalu teknis. Tabel boleh dipakai untuk angka.
4. Semua angka di paper harus berasal dari file di `results/`. Jangan mengarang hasil, referensi, atau angka. Kalau hasil belum ada, tulis placeholder yang jelas seperti `\textbf{[TODO: akurasi GASF]}`.
5. Kutip paper acuan setiap kali memakai metode atau membandingkan hasil dengannya.
6. Temuan soal MTF Normal boleh dibahas sebagai pengamatan, tanpa nada mengkritik paper acuan. Isu kebocoran data tidak perlu dibahas panjang.
7. Jangan pakai em dash.
8. Pastikan paper berhasil dikompilasi. `latexmk` tidak ada di environment, gunakan `pdflatex` dan `bibtex` (pdflatex, bibtex, pdflatex, pdflatex). Kalau gagal, beri tahu user.

## Cara bekerja

1. Kerjakan bertahap dan berhenti untuk melapor setelah tiap tahap: data dan EDA, encoding, training, analisis dan interpretasi, paper.
2. Setelah tiap tahap, jelaskan hasilnya dengan bahasa sederhana: apa yang terlihat di figur dan apa artinya. Penjelasan ini akan dipakai user untuk memahami dan menulis ulang insight.
3. Kalau ada hasil yang aneh (akurasi 100% di semua encoding, kelas tidak seimbang, kebocoran data antara train dan test), laporkan dan selidiki penyebabnya, jangan disembunyikan.
