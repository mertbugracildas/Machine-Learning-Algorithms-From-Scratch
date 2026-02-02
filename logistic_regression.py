import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#dosya yükleme ayırma kısmı
file_name = r'C:\Users\KMU\Desktop\Yeni klasör\exam_score.csv'
data = pd.read_csv(file_name, header=None)
data.columns = ['Exam_1_Score', 'Exam_2_Score', 'Admitted']
X = data[['Exam_1_Score', 'Exam_2_Score']].values
y = data['Admitted'].values.reshape(-1, 1)
#görselleştirme, grafiği oluşturma
admitted_indices = np.where(y == 1)[0]
not_admitted_indices = np.where(y == 0)[0]
plt.figure(figsize=(10, 6))
plt.scatter(X[admitted_indices, 0], X[admitted_indices, 1], c='green', marker='+', label='Kabul Edildi (1)')
plt.scatter(X[not_admitted_indices, 0], X[not_admitted_indices, 1], c='red', marker='o', label='Kabul Edilmedi (0)')
plt.xlabel('Sınav 1 Puanı')
plt.ylabel('Sınav 2 Puanı')
plt.title('Sınav Puanlarına Göre Kabul Durumu')
plt.legend()
plt.grid(True)
plt.savefig('acceptance_scatter_plot.png')
plt.close()
#normalizasyon
mu = np.mean(X, axis=0)
sigma = np.std(X, axis=0)
X_normalized = (X - mu) / sigma
m = X_normalized.shape[0]
X_final = np.concatenate((np.ones((m, 1)), X_normalized), axis=1)
initial_theta = np.zeros((X_final.shape[1], 1))
#lojistik regresyon, olasılık tahmini
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
#maliyet fonksiyonu, modelin hata payını hesaplar
def cost_function(X, y, theta):
    m = y.shape[0]
    h = sigmoid(np.dot(X, theta))
    J = (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return J
#Gradyan Fonksiyonu,maliyeti azlatmak için  theta yönünü hesaplar
def gradient(X, y, theta):
    m = y.shape[0]
    h = sigmoid(np.dot(X, theta))
    error = h - y
    grad = (1/m) * np.dot(X.T, error)
    return grad
#gradyan inişi, ağırlıkları optimize eden öğrenme kısmı
def gradient_descent(X, y, theta, alpha, num_iters):
    J_history = []
    for i in range(num_iters):
        grad = gradient(X, y, theta)
        theta = theta - alpha * grad
        J_history.append(cost_function(X, y, theta))
    return theta, J_history
#eğitim
alpha = 0.01
num_iters = 1500
theta = np.zeros((X_final.shape[1], 1))
final_theta, J_history = gradient_descent(X_final, y, theta, alpha, num_iters)
#maliyet grafiği
plt.figure(figsize=(10, 6))
plt.plot(range(num_iters), J_history, color='blue')
plt.xlabel('İterasyon Sayısı')
plt.ylabel('Maliyet J(theta)')
plt.title('Gradyan İnişi: Maliyetin İterasyonla Azalması')
plt.grid(True)
plt.savefig('cost_history.png')
plt.close()
#tahmin fonksiyonu
def predict(X, theta):
    h = sigmoid(np.dot(X, theta))
    predictions = np.where(h >= 0.5, 1, 0)
    return predictions
#modelin doğruluğu
y_pred = predict(X_final, final_theta)
accuracy = np.sum(y_pred == y) / y.shape[0]
#not girişi tahmin sistemi
new_scores = np.array([[50, 75]])
X_new_normalized = (new_scores - mu) / sigma
X_new_final = np.concatenate((np.ones((1, 1)), X_new_normalized), axis=1)
prob = sigmoid(np.dot(X_new_final, final_theta))[0][0]
prediction_result = "Kabul Edildi (1)" if prob >= 0.5 else "Kabul Edilmedi (0)"
print(f"Sınav Puanları: Sınav 1={new_scores[0, 0]}, Sınav 2={new_scores[0, 1]}, Kabul Olasılığı: {prob * 100:.2f}%")
print(f"Karar: {prediction_result}")