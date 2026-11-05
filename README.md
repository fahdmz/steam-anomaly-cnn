# Deteksi Anomali Jumlah Pemain Game Steam dengan Image Encoding dan CNN

Tugas kelompok visualisasi data time series menggunakan Convolutional Neural Network (CNN), Institut Teknologi Bandung.

**Anggota kelompok**
- Fahd Muhammad Zahid (13524078)
- Ishaq Irfan Farizal (13524094)

## Tentang proyek

Proyek ini mengikuti metode Rahadian dkk. (2023): data time series dipotong menjadi segmen, setiap segmen diubah menjadi gambar dengan teknik image encoding, lalu gambar tersebut diklasifikasi oleh CNN sebagai normal atau anomali. Bedanya, kami memakai data jumlah pemain harian enam game di Steam, bukan data tegangan motor DC seperti pada paper acuan. Kami juga menambahkan visualisasi Grad-CAM untuk melihat bagian waktu mana yang diperhatikan CNN.

Pertanyaan yang ingin kami jawab:
1. Encoding mana yang paling membantu CNN mengenali anomali pada data jumlah pemain game?
2. Apakah temuan paper acuan (selisih standar deviasi paling berpengaruh terhadap akurasi) tetap berlaku pada data ini?
3. Apa yang bisa dipelajari dari data dan dari cara CNN "melihat" gambar hasil encoding?

## Data

Data diambil dari repositori [ibrhr/Steam-Game-Analytics](https://github.com/ibrhr/Steam-Game-Analytics), yang bersumber dari SteamDB. Game yang dipakai adalah Dark Souls 3, Euro Truck Simulator 2, Garry's Mod, Rainbow Six Siege, Stardew Valley, dan Terraria.

Data dibersihkan, diseragamkan ke resolusi harian, lalu dipotong menjadi 1.229 segmen berisi 20 hari. Untuk setiap segmen dibuat salinan anomali dengan menyuntikkan lonjakan (2 sampai 5 kali lipat) atau penurunan (seperdua sampai seperlima) pada 1 sampai 5 hari acak.

## Metode singkat

1. Transformasi log10 dan standarisasi per game.
2. 14 image encoding dengan pustaka pyts: GADF, GASF, serta MTF dengan strategi uniform, quantile, dan normal (n_bins 2, 4, 5, 10).
3. CNN kecil yang sama untuk semua encoding, sesuai arsitektur paper acuan.
4. Perhitungan Δmean, ΔSD, dan Δcount antara gambar normal dan anomali, lalu korelasi Pearson terhadap akurasi CNN.
5. Grad-CAM pada model GADF untuk beberapa contoh segmen anomali.

## Hasil utama

| Keluarga encoding | Akurasi test |
|---|---|
| GADF dan GASF | 97,3% sampai 97,8% |
| MTF Uniform | 82,2% sampai 92,7% |
| MTF Normal | 75,4% sampai 80,5% |
| MTF Quantile | 56,5% sampai 70,8% |

Korelasi ΔSD terhadap akurasi pada data kami adalah r = 0,507, jauh lebih lemah dibanding r = 0,944 pada paper acuan.

Beberapa insight yang kami temukan:
- Jumlah pemain pada akhir pekan konsisten lebih tinggi daripada hari kerja di keenam game.
- GADF dan GASF paling baik, dan Grad-CAM menunjukkan CNN memperhatikan hari yang memang disuntik anomali.
- MTF Quantile lemah karena hanya memperhatikan urutan nilai, bukan besarnya lonjakan.
- MTF Normal tidak gagal seperti pada paper acuan karena data kami sudah distandarisasi, walaupun CNN sebagian memakai "jalan pintas" dari gambar yang berwarna polos.
- Lonjakan asli pada data (misalnya saat update atau diskon) membuat sebagian anomali buatan sulit dikenali.

Pembahasan lengkap ada di paper pada folder `paper/`.

## Struktur folder

```
data/raw/        data asli per game
data/processed/  segmen dan label
notebooks/       notebook per tahap dan notebook lengkap (00_lengkap.ipynb)
figures/         semua figur
results/         tabel hasil (CSV)
paper/           paper LaTeX
```

## Cara menjalankan

Proyek ini dibuat dengan Python 3.13.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/00_lengkap.ipynb
```

Notebook `00_lengkap.ipynb` mencakup seluruh alur, dari data sampai insight. Bobot model tidak disertakan di repositori karena ukurannya besar, jadi untuk menjalankan dari awal, ubah `LATIH_ULANG = True` pada sel konfigurasi. Training ulang bisa menghasilkan angka yang sedikit berbeda dari yang tertulis di paper.

Untuk mengompilasi paper:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Referensi

- H. Rahadian, S. Bandong, A. Widyotriatmo, dan E. Joelianto, "Image encoding selection based on Pearson correlation coefficient for time series anomaly detection," *Alexandria Engineering Journal*, vol. 82, hlm. 304 sampai 322, 2023.
- Z. Wang dan T. Oates, "Imaging time-series to improve classification and imputation," *IJCAI*, 2015.
- J. Faouzi dan H. Janati, "pyts: A Python Package for Time Series Classification," *Journal of Machine Learning Research*, vol. 21, 2020.
- R. R. Selvaraju dkk., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization," *ICCV*, 2017.