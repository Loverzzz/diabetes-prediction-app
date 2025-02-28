# Pipeline Prediksi Diabetes Menggunakan Machine Learning

Proyek ini menggunakan TensorFlow Extended (TFX) untuk membangun pipeline machine learning end-to-end dalam memprediksi kemungkinan seseorang mengalami diabetes berdasarkan data historis. Pipeline ini terdiri dari beberapa tahapan, termasuk pengambilan data, validasi, pra-pemrosesan, pelatihan model, evaluasi, dan penyebaran model.

## 1. Informasi Dataset

Dataset yang digunakan dalam proyek ini adalah dataset prediksi diabetes yang berisi data medis pasien, dengan fitur seperti usia, jenis kelamin, tekanan darah, kadar glukosa darah, dan informasi lainnya yang digunakan untuk memprediksi apakah seseorang menderita diabetes atau tidak.

Dataset ini disimpan dalam format CSV dan terdapat dalam folder `data/diabetes_prediction_dataset.csv`. Dataset ini memiliki kolom target yang berisi label **diabetes** yang menunjukkan apakah seseorang menderita diabetes (1) atau tidak (0).

## 2. Persoalan yang Ingin Diselesaikan

Persoalan yang ingin diselesaikan adalah **prediksi kemungkinan seseorang mengidap diabetes**. Dengan menggunakan data medis yang tersedia, tujuan utama dari model ini adalah untuk memberikan prediksi yang akurat, yang dapat membantu dalam deteksi dini penyakit diabetes, sehingga tindakan pencegahan atau pengobatan dapat segera dilakukan.

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

### Metrik Evaluasi:
Beberapa metrik evaluasi yang digunakan untuk menilai performa model adalah:
- **AUC (Area Under Curve)**: Mengukur seberapa baik model memisahkan dua kelas.
- **Precision**: Mengukur seberapa tepat prediksi positif yang dihasilkan model.
- **Recall**: Mengukur seberapa banyak data positif yang dapat terdeteksi oleh model.
- **F1-Score**: Harmonik rata-rata antara precision dan recall.
- **Accuracy**: Proporsi prediksi yang benar dari total prediksi.

## 5. Performa Model

Model yang dihasilkan dari pipeline ini dievaluasi dengan menggunakan **Evaluator** dan berbagai metrik, di antaranya AUC dan F1-Score. Berdasarkan hasil evaluasi, model ini menunjukkan performa yang baik dengan AUC tinggi dan nilai F1-Score yang memadai, menunjukkan keseimbangan antara precision dan recall.

## 6. Model Deployment

Model yang telah dilatih akan di-deploy untuk digunakan dalam aplikasi berbasis web menggunakan **Railway** sebagai platform penyebaran. Railway memungkinkan pengelolaan dan skalabilitas aplikasi dengan mudah, sehingga model dapat diakses untuk prediksi secara real-time.

### Platform yang Digunakan:
- **Railway**: Platform cloud untuk deployment model yang memungkinkan hosting dan pengelolaan aplikasi secara otomatis.

## 7. Tautan Web App untuk Mengakses Model Serving

Model yang telah di-deploy dapat diakses melalui web app dengan tautan berikut:  
[**Diabetes Prediction Web App**]([http://diabetes-prediction-app.railway.app](https://diabetes-prediction-diabetes-prediction.up.railway.app/))

Tautan ini memungkinkan pengguna untuk mengunggah data baru dan menerima prediksi tentang kemungkinan mengidap diabetes.

## 8. Hasil Monitoring

Penggunaan **Prometheus** dan **Grafana** memungkinkan pemantauan performa model secara real-time, dengan metrik seperti **jumlah permintaan** dan **latensi model**. Hasil monitoring menunjukkan bahwa model berfungsi dengan baik, dengan jumlah permintaan yang terus meningkat dan latensi yang stabil, yang menunjukkan efisiensi model dalam melakukan prediksi.

Gambar berikut menunjukkan grafik monitoring untuk jumlah permintaan prediksi:

![monitoring 2](https://github.com/user-attachments/assets/322fe059-38ee-4916-9fa6-e59d34037564)

![monitoring 1](https://github.com/user-attachments/assets/1c44f806-6c47-4311-8015-3aa6d302a966)

## Kesimpulan

Proyek ini menyediakan pipeline yang robust untuk memprediksi diabetes menggunakan machine learning. Pipeline ini dapat dikembangkan lebih lanjut dengan menambahkan teknik pra-pemrosesan tambahan, tipe model lain, atau kemampuan monitoring untuk meningkatkan kinerja dan kegunaan.
