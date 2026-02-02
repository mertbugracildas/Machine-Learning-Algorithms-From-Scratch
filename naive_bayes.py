#231312055 Mert Buğra ÇİLDAŞ yapay zeka kullandım hocam
import math
#veri seti
training_data = [
    ("maç çok heyecanlıydı", "spor"),
    ("hakem penaltı verdi", "spor"),
    ("gol kralı belli oldu", "spor"),
    ("seçim sonuçları açıklandı", "siyaset"),
    ("meclis yeni yasa çıkardı", "siyaset"),
    ("oy oranları yükseldi", "siyaset"),
    ("maçta çok gol oldu", "spor"),
    ("bakan açıklama yaptı", "siyaset")
]

test_sentences = ["hakem maçı bitirdi", "yeni seçim yasası", "bugün hava çok güzel", "bugünkü oturum çok sert geçti", "beraberlik iyi sonuç"]#test ettiğimiz cümleler
#eğitimi burada yapıyoruz
class_doc_counts = {"spor": 0, "siyaset": 0} #bu kelimelerden kaç tane var
word_presence_counts = {"spor": {}, "siyaset": {}} #kaç cümlede geçti
vocab = set()

for sentence, category in training_data:
    class_doc_counts[category] += 1
    unique_words = set(sentence.lower().split()) # multinomial den kurtardığım kısım

    for word in unique_words:
        vocab.add(word)
        if word not in word_presence_counts[category]:
            word_presence_counts[category][word] = 0
        word_presence_counts[category][word] += 1

total_docs = sum(class_doc_counts.values())

print(f"Model Eğitildi. (Toplam Cümle: {total_docs}, Kelime Haznesi: {len(vocab)})")
#tahmin kısmı
def predict(sentence):
    words = set(sentence.lower().split())
    scores = {}

    for category in class_doc_counts:
        scores[category] = math.log(class_doc_counts[category] / total_docs)

        for word in words:
            count = word_presence_counts[category].get(word, 0)
            #lablas yumuşatması burada yapıyoruz
            prob = (count + 1) / (class_doc_counts[category] + 2)

            scores[category] += math.log(prob)

    return max(scores, key=scores.get)
#sonuç çıktı
print("-" * 30)
for sentence in test_sentences:
    result = predict(sentence)
    print(f"Cümle: {sentence:<20} -> Tahmin: {result}")