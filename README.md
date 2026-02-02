# Machine Learning Algorithms Implementation / Makine Öğrenmesi Uygulamaları

This repository contains implementations of fundamental Machine Learning algorithms using Python. The primary objective is to demonstrate the mathematical foundations behind these models by implementing them from scratch, without relying on high-level libraries for the core logic.

Bu depo, temel Makine Öğrenmesi algoritmalarının Python kullanılarak yapılmış uygulamalarını içerir. Temel amaç, çekirdek mantık için hazır kütüphanelere güvenmek yerine, bu modelleri sıfırdan uygulayarak arkalarındaki matematiksel temelleri göstermektir.

---

## 🇬🇧 Project 1: Logistic Regression
This project implements a Logistic Regression model to predict the pass/fail status of a student based on two exam scores.

### Implementation Details
The model is built using a mathematical approach rather than pre-built library functions (like `sklearn.linear_model`). The following components were manually implemented:
* **Sigmoid Function:** Implemented to map predictions to probability values between 0 and 1.
* **Cost Function:** Calculated to measure the accuracy of the model during training.
* **Gradient Descent:** Applied to optimize parameters and minimize the cost function iteratively.

### Technologies & Dataset
* **Tech:** Python, Pandas, NumPy, Matplotlib
* **Data:** `exam_score.csv` contains two exam scores and a binary target variable (0: Fail, 1: Pass).

---

## 🇹🇷 Proje 1: Lojistik Regresyon
Bu proje, iki sınav sonucuna dayanarak bir öğrencinin dersi geçme veya kalma durumunu tahmin etmek amacıyla Lojistik Regresyon modelini uygular.

### Uygulama Detayları
Model, `sklearn` gibi hazır kütüphane fonksiyonları yerine, algoritmanın matematiksel altyapısı kodlanarak oluşturulmuştur. Aşağıdaki bileşenler manuel olarak (from scratch) uygulanmıştır:
* **Sigmoid Fonksiyonu:** Tahmin çıktılarını 0 ile 1 arasında bir olasılık değerine dönüştürmek için kullanıldı.
* **Maliyet (Cost) Fonksiyonu:** Modelin eğitim sürecindeki hata payını ölçmek için hesaplandı.
* **Gradient Descent:** Parametreleri optimize etmek ve hatayı yinelemeli (iterative) olarak minimize etmek için uygulandı.

### Teknolojiler ve Veri Seti
* **Teknolojiler:** Python, Pandas, NumPy, Matplotlib
* **Veri:** `exam_score.csv` dosyası, iki sınav notunu ve öğrencinin başarı durumunu (0: Kaldı, 1: Geçti) içerir.
