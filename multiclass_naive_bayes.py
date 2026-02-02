import math

#VERİLER
X_train = [
    ["Az", "Lisans", "Yok"], ["Az", "Lisans", "Var"],
    ["Orta", "Lisans", "Yok"], ["Orta", "Yüksek", "Var"],
    ["Çok", "Yüksek", "Yok"], ["Çok", "Doktora", "Var"],
    ["Az", "Yüksek", "Yok"], ["Orta", "Doktora", "Yok"],
    ["Çok", "Lisans", "Var"], ["Az", "Doktora", "Var"]
]

y_train = ["RED", "MULAKAT", "MULAKAT", "ISE_AL", "ISE_AL", "ISE_AL", "RED", "MULAKAT", "ISE_AL", "MULAKAT"]
#test verilerimiz
test_data = [
    ["Çok", "Lisans", "Yok"],
    ["Az", "Yüksek", "Yok"],
    ["Az", "Lisans", "Yok"],
    ["Az", "Doktora", "Yok"],
    ["Orta", "Yüksek", "Var"]
]

#EĞİTİM
class_counts = {} # Her sınıftan kaç örnek var
feature_counts = {} # hangi özellik kaç kere geçmiş?
class_total_features = {} #toplam özellik sayısı
vocab = set()

for features, category in zip(X_train, y_train):
    if category not in class_counts:
        class_counts[category] = 0
        feature_counts[category] = {}
        class_total_features[category] = 0

    class_counts[category] += 1

    # Özellikleri sayma
    for feat in features:
        vocab.add(feat)
        if feat not in feature_counts[category]:
            feature_counts[category][feat] = 0
        feature_counts[category][feat] += 1
        class_total_features[category] += 1

total_docs = sum(class_counts.values())
vocab_size = len(vocab)
print(f"Model Eğitildi. Sınıflar: {list(class_counts.keys())}\n")

#TAHMİN FONKSİYONU
def predict(features):
    scores = {}

    for category in class_counts:
        scores[category] = math.log(class_counts[category] / total_docs)

        #Multinomial Formülü
        for feat in features:
            count = feature_counts[category].get(feat, 0)

            # Laplace yumuşatması
            prob = (count + 1) / (class_total_features[category] + vocab_size)

            scores[category] += math.log(prob)

    return max(scores, key=scores.get)

#SONUÇ
print("-" * 45)
print(f"{'ÖZELLİKLER':<30} | {'TAHMİN'}")
print("-" * 45)

for instance in test_data:
    result = predict(instance)

    print(f"{str(instance):<30} | {result}")
