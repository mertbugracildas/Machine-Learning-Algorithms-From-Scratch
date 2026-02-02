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

## 🇬🇧 Project 3: Naive Bayes Classifier (Text Classification)

This project implements the Naive Bayes algorithm specifically for **text classification** tasks (Natural Language Processing basics).

### Implementation Details
The model predicts the category of a given sentence (e.g., Sports vs. Politics) based on word frequencies.
* **Bayes' Theorem:** Calculates the posterior probability $P(c|x)$ using the prior probability $P(c)$ and likelihood $P(x|c)$.
* **Bag of Words:** The model analyzes the frequency of words in sentences to determine which category they belong to.
* **Binary Classification:** Classifies inputs into two distinct categories by comparing calculated probabilities.

### Technologies & Dataset
* **Tech:** Python, NumPy.
* **Data:** `tekli.txt` (Training) and `teskli_test.txt` (Testing) containing sentences and their categories.

---

## 🇹🇷 Proje 3: Naive Bayes Sınıflandırıcısı (Metin Sınıflandırma)

Bu proje, özellikle **metin sınıflandırma** (Doğal Dil İşleme temelleri) görevleri için Naive Bayes algoritmasını uygular.

### Uygulama Detayları
Model, kelime sıklıklarına dayanarak verilen bir cümlenin kategorisini (Örn: Spor veya Siyaset) tahmin eder.
* **Bayes Teoremi:** Önsel (prior) ve olabilirlik (likelihood) değerlerini kullanarak bir cümlenin belirli bir kategoriye ait olma olasılığını hesaplar.
* **Kelime Çantası (Bag of Words):** Model, cümlelerin hangi kategoriye ait olduğunu belirlemek için kelimelerin geçiş sıklığını analiz eder.
* **İkili Sınıflandırma:** Hesaplanan olasılıkları karşılaştırarak girdileri iki farklı kategoriye ayırır.

### Teknolojiler ve Veri Seti
* **Teknolojiler:** Python, NumPy.
* **Veri:** Cümleleri ve kategorilerini içeren `tekli.txt` (Eğitim) ve `teskli_test.txt` (Test) dosyaları.

---

## 🇬🇧 Project 4: Multi-Class Naive Bayes Classifier

This project extends the Naive Bayes algorithm to handle multi-class classification problems, where data with categorical features needs to be categorized into more than two groups.

### Implementation Details
Similar to the binary version, this model calculates the probability of a data point belonging to each possible class and assigns it to the class with the highest probability.
* **Multi-Class Logic:** Instead of just $P(Class A)$ vs $P(Class B)$, the model computes posterior probabilities for $C_1, C_2, ..., C_n$ and selects the maximum.
* **Categorical Features:** The likelihood is calculated by analyzing categorical attributes (e.g., "Experience Level", "Education") assuming feature independence.

### Technologies & Dataset
* **Tech:** Python, NumPy.
* **Data:** `coklu.txt` (Training) and `coklu_test.txt` (Testing) containing categorical features and labels for multiple classes.

---

## 🇹🇷 Proje 4: Çok Sınıflı (Multi-Class) Naive Bayes

Bu proje, kategorik özelliklere sahip verilerin ikiden fazla gruba ayrılması gereken durumlar için Naive Bayes algoritmasını genişletir.

### Uygulama Detayları
İkili versiyona benzer şekilde, bu model bir veri noktasının olası her bir sınıfa ait olma olasılığını hesaplar ve en yüksek olasılığa sahip olan sınıfı atar.
* **Çok Sınıflı Mantık:** Sadece A veya B sınıfı yerine, model $C_1, C_2, ..., C_n$ sınıfları için sonsal olasılıkları hesaplar ve maksimum olanı seçer.
* **Kategorik Özellikler:** Olabilirlik (Likelihood), özelliklerin bağımsız olduğu varsayılarak kategorik niteliklerin (Örn: "Tecrübe", "Eğitim") analiziyle hesaplanır.

### Teknolojiler ve Veri Seti
* **Teknolojiler:** Python, NumPy.
* **Veri:** `coklu.txt` (Eğitim) ve `coklu_test.txt` (Test) dosyaları, birden fazla sınıf için etiketleri içerir.

---

## 5. K-Means Clustering (using Scikit-Learn)

### 🇬🇧 Description
Unlike the previous implementations built from scratch, this project utilizes the industry-standard **Scikit-Learn** library.
* **Objective:** To demonstrate familiarity with professional machine learning tools used in real-world applications.
* **Library:** `sklearn.cluster.KMeans` used for optimizing data grouping.

### 🇹🇷 Açıklama
Sıfırdan (from scratch) geliştirilen önceki uygulamaların aksine, bu projede endüstri standardı olan **Scikit-Learn** kütüphanesi kullanılmıştır.
* **Amaç:** Gerçek dünya uygulamalarında kullanılan profesyonel makine öğrenmesi araçlarına olan hakimiyeti göstermektir.
* **Kütüphane:** Veri gruplandırmasını optimize etmek için `sklearn.cluster.KMeans` kullanılmıştır.
