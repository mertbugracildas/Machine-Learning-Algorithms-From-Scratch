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

---

## 🇬🇧 Project 2: Linear Regression
This project implements a Linear Regression model to predict a continuous value (dependent variable) based on an independent variable.

### Implementation Details
The algorithm finds the "best fit line" for the given dataset by minimizing the error.
* **Model:** $y = mx + b$ (Equation of a line).
* **Cost Function (MSE):** Calculated to measure the average squared difference between the estimated values and the actual value.
* **Gradient Descent:** Used to update the weights ($m$) and bias ($b$) to reach the minimum error.

### Technologies & Dataset
* **Tech:** Python, Matplotlib (for plotting the regression line).
* **Data:** `dataset.txt` contains the data points used for training the model.

---

## 🇹🇷 Proje 2: Doğrusal Regresyon (Linear Regression)
Bu proje, bağımsız bir değişkene dayanarak sürekli bir değeri (bağımlı değişken) tahmin etmek için Doğrusal Regresyon modelini uygular.

### Uygulama Detayları
Algoritma, hatayı minimize ederek verilen veri seti için "en iyi uyan doğruyu" (best fit line) bulur.
* **Model:** $y = mx + b$ (Doğru denklemi).
* **Maliyet Fonksiyonu (MSE):** Tahmin edilen değerler ile gerçek değerler arasındaki karesel farkların ortalamasını ölçmek için hesaplandı.
* **Gradient Descent:** Hatayı minimuma indirmek için ağırlıkları ($m$) ve sapmayı ($b$) güncellemek amacıyla kullanıldı.

### Teknolojiler ve Veri Seti
* **Teknolojiler:** Python, Matplotlib (regresyon doğrusunu çizdirmek için).
* **Veri:** `dataset.txt` dosyası, modeli eğitmek için kullanılan veri noktalarını içerir.

---

## 🇬🇧 Project 3: Naive Bayes Classifier
This project implements the Naive Bayes algorithm, a probabilistic classifier based on Bayes' Theorem, specifically for binary classification tasks.

### Implementation Details
The model predicts the class of a given data point by calculating the probability of it belonging to each class.
* **Bayes' Theorem:** Calculates the posterior probability $P(c|x)$ using the prior probability $P(c)$ and likelihood $P(x|c)$.
  * Formula: $P(c|x) = \frac{P(x|c) \cdot P(c)}{P(x)}$
* **Binary Classification:** The model classifies inputs into two distinct categories (e.g., 0 or 1) by comparing the calculated probabilities.
* **Gaussian Distribution:** (If used in code) Assumes that the continuous values associated with each class are distributed according to a Gaussian (Normal) distribution.

### Technologies & Dataset
* **Tech:** Python, NumPy (for probabilistic calculations).
* **Data:** `tekli.txt` (Training) and `teskli_test.txt` (Testing) datasets containing features and binary class labels.

---

## 🇹🇷 Proje 3: Naive Bayes Sınıflandırıcısı
Bu proje, ikili sınıflandırma (binary classification) görevleri için Bayes Teoremi'ne dayanan olasılıksal bir sınıflandırıcı olan Naive Bayes algoritmasını uygular.

### Uygulama Detayları
Model, verilen bir veri noktasının her bir sınıfa ait olma olasılığını hesaplayarak tahminleme yapar.
* **Bayes Teoremi:** Önsel olasılık (prior) $P(c)$ ve olabilirlik (likelihood) $P(x|c)$ değerlerini kullanarak sonsal olasılığı (posterior) $P(c|x)$ hesaplar.
* **İkili Sınıflandırma:** Model, hesaplanan olasılıkları karşılaştırarak girdileri iki farklı kategoriye (örneğin 0 veya 1) ayırır.
* **Gauss Dağılımı:** (Kodda kullanıldıysa) Her bir sınıfla ilişkili sürekli değerlerin bir Gauss (Normal) dağılımına uyduğunu varsayar.

### Teknolojiler ve Veri Seti
* **Teknolojiler:** Python, NumPy (olasılık hesaplamaları için).
* **Veri:** Özellikleri ve ikili sınıf etiketlerini içeren `tekli.txt` (Eğitim) ve `teskli_test.txt` (Test) dosyaları.

---

## 🇬🇧 Project 4: Multi-Class Naive Bayes Classifier
This project extends the Naive Bayes algorithm to handle multi-class classification problems, where the data needs to be categorized into more than two groups.

### Implementation Details
Similar to the binary version, this model calculates the probability of a data point belonging to each possible class and assigns it to the class with the highest probability.
* **Multi-Class Logic:** Instead of just $P(Class A)$ vs $P(Class B)$, the model computes posterior probabilities for $C_1, C_2, ..., C_n$ and selects the maximum: $\hat{y} = \arg\max_{k} P(C_k | x)$.
* **Handling Multiple Features:** The likelihood is calculated by multiplying the probabilities of individual features (assuming independence).

### Technologies & Dataset
* **Tech:** Python, NumPy.
* **Data:** `coklu.txt` (Training) and `coklu_test.txt` (Testing) containing features and labels for multiple classes (e.g., Class 0, Class 1, Class 2).

---

## 🇹🇷 Proje 4: Çok Sınıflı (Multi-Class) Naive Bayes
Bu proje, Naive Bayes algoritmasını ikiden fazla kategoriye ayrılması gereken veri setleri için genişletir (Çok Sınıflı Sınıflandırma).

### Uygulama Detayları
İkili versiyona benzer şekilde, bu model bir veri noktasının olası her bir sınıfa ait olma olasılığını hesaplar ve en yüksek olasılığa sahip olan sınıfı atar.
* **Çok Sınıflı Mantık:** Sadece A veya B sınıfı yerine, model $C_1, C_2, ..., C_n$ sınıfları için sonsal olasılıkları hesaplar ve maksimum olanı seçer: $\hat{y} = \arg\max_{k} P(C_k | x)$.
* **Çoklu Özellik Yönetimi:** Olabilirlik (Likelihood), özelliklerin bağımsız olduğu varsayılarak tek tek olasılıkların çarpımıyla hesaplanır.

### Teknolojiler ve Veri Seti
* **Teknolojiler:** Python, NumPy.
* **Veri:** `coklu.txt` (Eğitim) ve `coklu_test.txt` (Test) dosyaları, birden fazla sınıf (Örn: Sınıf 0, Sınıf 1, Sınıf 2) için etiketleri içerir.

---

## 5. K-Means Clustering (using Scikit-Learn)

**🇬🇧 Description:**
Unlike the previous implementations built from scratch, this project utilizes the industry-standard **Scikit-Learn** library.
* **Objective:** To demonstrate familiarity with professional machine learning tools used in real-world applications.
* **Library:** `sklearn.cluster.KMeans` used for optimizing data grouping.

**🇹🇷 Açıklama:**
Sıfırdan (from scratch) geliştirilen önceki uygulamaların aksine, bu projede endüstri standardı olan **Scikit-Learn** kütüphanesi kullanılmıştır.
* **Amaç:** Gerçek dünya uygulamalarında kullanılan profesyonel makine öğrenmesi araçlarına olan hakimiyeti göstermektir.
* **Kütüphane:** Veri gruplandırmasını optimize etmek için `sklearn.cluster.KMeans` kullanılmıştır.
