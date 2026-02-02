# 231312055 Mert Buğra Çildaş - Lojistik Regresyon yapay zeka ile düzenlendi.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Dosya Yükleme ve Veri Ayırma
file_name = 'dataset.txt'
data = pd.read_csv(file_name, header=None)
data.columns = ['nufus', 'kar']
X = data[['nufus']].values
y = data[['kar']].values
m = y.shape[0] 

#Veri Görselleştirme
plt.figure(figsize=(10, 6))
plt.scatter(X, y, c='red', marker='x', label='Eğitim Verisi')
plt.xlabel('Nüfus (x10.000)')
plt.ylabel('Kâr (x10.000)')
plt.title('Nüfusa Göre Kâr Dağılımı')
plt.legend()
plt.grid(True)
plt.savefig('linear_scatter_plot.png')
plt.close()

# Maliyet Fonksiyonu
def compute_cost(X_b, y, theta):
    m = y.shape[0]
    h = np.dot(X_b, theta) 
    error = h - y
    J = (1 / (2 * m)) * np.sum(error**2)
    return J
# Gradyan İnişi Fonksiyonu
def gradient_descent(X_b, y, theta, alpha, num_iters):
    m = y.shape[0]
    J_history = [] 
    for i in range(num_iters):
        h = np.dot(X_b, theta)
        error = h - y
        grad = (1 / m) * np.dot(X_b.T, error)
        theta = theta - alpha * grad
        J_history.append(compute_cost(X_b, y, theta))

    return theta, J_history

#Maliyet Hesaplamaları (Normalizasyonsuz)
X_b_raw = np.concatenate((np.ones((m, 1)), X), axis=1) 
theta_zero = np.zeros((2, 1))
cost_zero = compute_cost(X_b_raw, y, theta_zero)
print(f"Theta = [0, 0] için başlangıç maliyeti (J(theta)): {cost_zero:.2f}")
theta_test = np.array([[-1], [2]])
cost_test = compute_cost(X_b_raw, y, theta_test)
print(f"Theta = [-1, 2] için test maliyeti: {cost_test:.2f}")
print("---")

# Normalizasyon
mu = np.mean(X, axis=0)
sigma = np.std(X, axis=0)
X_normalized = (X - mu) / sigma
X_final = np.concatenate((np.ones((m, 1)), X_normalized), axis=1)

# Modeli Eğitme
alpha = 0.01
num_iters = 1500
initial_theta = np.zeros((X_final.shape[1], 1))
final_theta, J_history = gradient_descent(X_final, y, initial_theta, alpha, num_iters)

print(f"Gradyan İnişi ile bulunan optimize edilmiş theta (normalize edilmiş veri için):")
print(f"Theta_0 (Intercept): {final_theta[0][0]:.4f}")
print(f"Theta_1 (Eğim): {final_theta[1][0]:.4f}")

# Maliyet Grafiği 
plt.figure(figsize=(10, 6))
plt.plot(range(num_iters), J_history, color='blue')
plt.xlabel('İterasyon Sayısı')
plt.ylabel('Maliyet J(theta)')
plt.title('Gradyan İnişi: Maliyetin İterasyonla Azalması [cite: 16]')
plt.grid(True)
plt.savefig('cost_history_linear.png')
plt.close()

# Regresyon Çizgisini Görselleştirme
predictions = np.dot(X_final, final_theta)
plt.figure(figsize=(10, 6))
plt.scatter(X, y, c='red', marker='x', label='Eğitim Verisi')
plt.plot(X, predictions, c='blue', label='Lineer Regresyon Çizgisi')
plt.xlabel('Nüfus (x10.000)')
plt.ylabel('Kâr (x10.000)')
plt.title('Lineer Regresyon Modeli Sonucu')
plt.legend()
plt.grid(True)
plt.savefig('linear_regression_fit.png')
plt.close()

# Tahmin Yapma 
new_populations = np.array([[3.5], [7.0]])
new_pop_normalized = (new_populations - mu) / sigma
new_pop_final = np.concatenate((np.ones((new_populations.shape[0], 1)), new_pop_normalized), axis=1)
new_predictions = np.dot(new_pop_final, final_theta)
print("---")
print(f"{new_populations[0][0]*10000:.0f} nüfus için kâr tahmini: {new_predictions[0][0]*10000:.2f} TL")
print(f"{new_populations[1][0]*10000:.0f} nüfus için kâr tahmini: {new_predictions[1][0]*10000:.2f} TL")