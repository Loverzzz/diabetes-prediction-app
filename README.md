# Pipeline Prediksi Diabetes Menggunakan Machine Learning

Proyek ini menggunakan TensorFlow Extended (TFX) untuk membangun pipeline machine learning end-to-end dalam memprediksi kemungkinan seseorang mengalami diabetes berdasarkan data historis. Pipeline ini terdiri dari beberapa tahapan, termasuk pengambilan data, validasi, pra-pemrosesan, pelatihan model, evaluasi, dan penyebaran model.

## 1. Informasi Dataset

Dataset yang digunakan dalam proyek ini adalah dataset prediksi diabetes yang berisi data medis pasien, dengan fitur seperti usia, jenis kelamin, tekanan darah, kadar glukosa darah, dan informasi lainnya yang digunakan untuk memprediksi apakah seseorang menderita diabetes atau tidak.

Dataset ini disimpan dalam format CSV dan terdapat dalam folder `data/diabetes_prediction_dataset.csv`. Dataset ini memiliki kolom target yang berisi label **diabetes** yang menunjukkan apakah seseorang menderita diabetes (1) atau tidak (0).

## 2. Persoalan yang Ingin Diselesaikan

**Masalah:** 

Diabetes adalah salah satu penyakit kronis yang paling umum di dunia, dengan dampak yang signifikan terhadap kesehatan masyarakat dan sistem kesehatan global. Menurut laporan dari Organisasi Kesehatan Dunia (WHO), lebih dari 422 juta orang di seluruh dunia menderita diabetes, dan angka ini terus meningkat setiap tahunnya. Diabetes juga menjadi penyebab kematian langsung bagi sekitar 1,5 juta orang setiap tahun. Penyakit ini sering kali berkembang tanpa gejala yang jelas pada tahap awal, sehingga banyak penderita baru menyadari kondisinya setelah komplikasi serius terjadi, seperti penyakit jantung, gagal ginjal, atau kebutaan.

**Urgensi **: 

Deteksi dini diabetes sangat penting untuk mencegah komplikasi yang lebih serius. Dengan mengetahui kemungkinan seseorang menderita diabetes lebih awal, langkah-langkah preventif seperti perubahan gaya hidup, pengaturan diet, dan intervensi medis dapat dilakukan untuk memperlambat atau bahkan mencegah perkembangan penyakit. Selain itu, deteksi dini juga dapat membantu mengurangi beban ekonomi pada sistem kesehatan, terutama di negara-negara dengan prevalensi diabetes yang tinggi.

Namun, deteksi dini diabetes sering kali menghadapi tantangan, seperti:

 1. Kurangnya akses ke fasilitas kesehatan di daerah terpencil.
 2. Ketergantungan pada pemeriksaan medis yang memerlukan biaya tinggi.
 3. Keterbatasan tenaga medis untuk menangani jumlah pasien yang terus meningkat.

**Solusi yang ditawarkan:** 

Persoalan yang ingin diselesaikan adalah **prediksi kemungkinan seseorang mengidap diabetes**. Dengan menggunakan data medis yang tersedia, tujuan utama dari model ini adalah untuk memberikan prediksi yang akurat, yang dapat membantu dalam deteksi dini penyakit diabetes, sehingga tindakan pencegahan atau pengobatan dapat segera dilakukan.

**Manfaat:** 

  1. Efisien: Prediksi dapat dilakukan dengan cepat tanpa memerlukan pemeriksaan medis yang rumit.
  2. Terjangkau: Menggunakan data yang sudah tersedia, seperti hasil pemeriksaan rutin, untuk membuat prediksi.
  3. Mudah Diakses: Model ini dapat diintegrasikan ke dalam aplikasi berbasis web, sehingga dapat digunakan oleh siapa saja, kapan saja, dan di mana saja.

## 3. Solusi Machine Learning

Solusi yang akan diterapkan dalam proyek ini adalah membangun model machine learning untuk mengklasifikasikan pasien apakah mereka menderita diabetes atau tidak, berdasarkan fitur medis yang ada. Model ini menggunakan **algoritma klasifikasi** dan diharapkan dapat mencapai kinerja yang baik pada metrik evaluasi seperti **AUC**, **Precision**, **Recall**, **F1-Score**, dan **Accuracy**.

### Target yang Ingin Dicapai:
- Model dapat memprediksi diabetes dengan akurasi tinggi.
- Model dapat di-deploy ke aplikasi berbasis web untuk digunakan oleh pengguna.

## 4. Metode Pengolahan Data, Arsitektur Model, dan Metrik Evaluasi

### Pengolahan Data:
Data yang digunakan akan diproses menggunakan beberapa tahapan:
1. **Validasi Data**: Validasi dilakukan untuk memastikan data sesuai dengan format yang diinginkan.
2. **Pra-pemrosesan**: Data akan diproses menggunakan komponen **Transform** untuk melakukan normalisasi dan transformasi fitur.
3. **Pembagian Data**: Dataset akan dibagi menjadi dua bagian: **train** (untuk pelatihan model) dan **eval** (untuk evaluasi model).

### Arsitektur Model:
Model yang digunakan dalam proyek ini adalah model **Klasifikasi** berbasis **Neural Network**. Model ini dibangun menggunakan TensorFlow dan dilatih untuk memprediksi apakah seseorang menderita diabetes atau tidak.

1. Input Layer
    Model menerima dua jenis input:
    - Fitur Kategorikal: Fitur ini diolah menggunakan representasi one-hot encoding dengan dimensi spesifik untuk setiap fitur. Input layer untuk fitur kategorikal dirancang menggunakan tf.keras.Input dengan dimensi (dim + 1), di mana dim adalah jumlah kategori unik dalam fitur tersebut.
    - Fitur Numerikal: Fitur ini diolah secara langsung sebagai input numerik dengan dimensi (1).
    Semua input ini kemudian digabungkan menggunakan layer Concatenate.
2. Hidden Layers
  Model memiliki tiga hidden layer Dense (fully connected layer) dengan konfigurasi sebagai berikut:

    - Layer 1:
      - Jumlah neuron: 256
      - Fungsi aktivasi: ReLU (Rectified Linear Unit)
      - Fungsi ini dipilih karena dapat menangani non-linearitas dengan baik dan mencegah saturasi gradien.
    - Layer 2:
      - Jumlah neuron: 64
      - Fungsi aktivasi: ReLU
    - Layer 3:
      - Jumlah neuron: 16
      - Fungsi aktivasi: ReLU
    Hidden layers ini dirancang untuk menangkap pola-pola kompleks dalam data, dengan jumlah neuron yang semakin berkurang untuk melakukan kompresi informasi.
 3. Output Layer
    - Jumlah neuron: 1
    - Fungsi aktivasi: Sigmoid
    - Fungsi ini dipilih karena model dirancang untuk klasifikasi biner, di mana output berupa probabilitas antara 0 dan 1.
  4. Optimizer
    - Model menggunakan Adam Optimizer dengan learning rate sebesar 0.001.
      Adam dipilih karena:
        - Menggabungkan keunggulan dari RMSProp dan Momentum.
        - Stabil dan cepat dalam proses konvergensi.
  5. Loss Function
    - Binary Crossentropy: Fungsi loss ini digunakan untuk klasifikasi biner, menghitung jarak antara prediksi probabilitas dan label sebenarnya.
  6. Metrics
    - Model dievaluasi menggunakan metrik:
      - BinaryAccuracy: Mengukur proporsi prediksi yang benar terhadap total prediksi.
  7. Training Configuration
      - Batch Size: 64
      - Epochs: 10
      - Callbacks: TensorBoard untuk memantau proses pelatihan model, seperti loss dan akurasi, secara visual.
  8. Training Process
     - Data Input: Data dimuat dari file TFRecord dengan format yang telah ditransformasi menggunakan TensorFlow Transform (TFT).
     - Training dan Validation: Dataset dibagi menjadi data latih dan validasi, masing-masing dengan batch size 64.
     - Steps per Epoch: Ditentukan oleh argumen fn_args.train_steps dan fn_args.eval_steps.
  9. Deployment
     - Model disimpan dalam format SavedModel dengan signature untuk serving, memungkinkan model digunakan dalam produksi melalui API. Fungsi serve_tf_examples_fn digunakan untuk parsing input selama serving.


### Metrik Evaluasi:
Beberapa metrik evaluasi yang digunakan untuk menilai performa model adalah:
- **Accuracy**: Proporsi prediksi yang benar dari total prediksi.
- **AUC (Area Under Curve)**: Mengukur seberapa baik model memisahkan dua kelas.
- **Precision**: Mengukur seberapa tepat prediksi positif yang dihasilkan model.
- **Recall**: Mengukur seberapa banyak data positif yang dapat terdeteksi oleh model.
- **F1-Score**: Harmonik rata-rata antara precision dan recall.

## 5. Performa Model

Model yang dihasilkan dari pipeline ini dievaluasi dengan menggunakan **Evaluator** dan berbagai metrik, di antaranya Accuracy sebesar 0.97 dan F1-Score sebesar 0.7934482025101043. Berdasarkan hasil evaluasi, model ini menunjukkan performa yang baik dengan Accuracy tinggi dan nilai F1-Score yang memadai, menunjukkan keseimbangan antara precision dan recall.

## 6. Model Deployment

Model yang telah dilatih akan di-deploy untuk digunakan dalam aplikasi berbasis web menggunakan **Railway** sebagai platform penyebaran. Railway memungkinkan pengelolaan dan skalabilitas aplikasi dengan mudah, sehingga model dapat diakses untuk prediksi secara real-time.

### Platform yang Digunakan:
- **Railway**: Platform cloud untuk deployment model yang memungkinkan hosting dan pengelolaan aplikasi secara otomatis.

## 7. Tautan Web App untuk Mengakses Model Serving

Model yang telah di-deploy dapat diakses melalui web app dengan tautan berikut:  
[**Diabetes Prediction Web App**] : [https://diabetes-prediction-diabetes-prediction.up.railway.app/](https://diabetes-prediction-diabetes-prediction.up.railway.app/v1/models/diabetes-classification-model/metadata)

Tautan ini memungkinkan pengguna untuk mengunggah data baru dan menerima prediksi tentang kemungkinan mengidap diabetes.

## 8. Hasil Monitoring

Penggunaan **Prometheus** dan **Grafana** memungkinkan pemantauan performa model secara real-time, dengan metrik seperti **jumlah permintaan** dan **latensi model**. Hasil monitoring menunjukkan bahwa model berfungsi dengan baik, dengan jumlah permintaan yang terus meningkat dan latensi yang stabil, yang menunjukkan efisiensi model dalam melakukan prediksi.

Gambar berikut menunjukkan grafik monitoring untuk jumlah permintaan prediksi:

![monitoring 2](https://github.com/user-attachments/assets/322fe059-38ee-4916-9fa6-e59d34037564)

![monitoring 1](https://github.com/user-attachments/assets/1c44f806-6c47-4311-8015-3aa6d302a966)

## Kesimpulan

Proyek ini menyediakan pipeline yang robust untuk memprediksi diabetes menggunakan machine learning. Pipeline ini dapat dikembangkan lebih lanjut dengan menambahkan teknik pra-pemrosesan tambahan, tipe model lain, atau kemampuan monitoring untuk meningkatkan kinerja dan kegunaan.
