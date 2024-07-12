import pandas as pd  # Veri işleme, CSV dosya girdi/çıktısı
from sklearn.model_selection import train_test_split  # Veri setini eğitim ve test setlerine bölme
from sklearn.preprocessing import StandardScaler # Verilerin ölçeklendirilmesi
from sklearn.metrics import accuracy_score  # Doğruluk metriği
from sklearn.naive_bayes import GaussianNB  # Gaussian Naive Bayes sınıflandırıcısı
import joblib  # Modeli kaydetmek/yüklemek için
from sklearn.utils import shuffle # Veri setini karıştırmak için

heart = pd.read_csv('heart.csv')

# Veri setini karıştırma
heart = shuffle(heart, random_state=0)

# Yinelenen satırları kaldırıyoruz
heart.drop_duplicates(keep='first', inplace=True)

# Özellikleri ve hedefi ayırıyoruz
x = heart.iloc[:,:-1].values
y = heart.iloc[:, -1].values

# Veriyi eğitim ve test setlerine ayırıyoruz (70% eğitim, 30% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Veriyi ölçeklendirme (Standart sapma ile ölçeklendirme)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Gaussian Naive Bayes modelini oluşturur
model = GaussianNB()
model.fit(x_train, y_train)  

# Test seti ile tahmin yapma işlemi
predicted = model.predict(x_test)

# Modelin doğruluğunu hesaplama ve yazdırma
print("Gaussian Naive Bayes Modelinin Doğruluğu: ", accuracy_score(y_test, predicted) * 100, "%")

# Modeli dosyaya kaydetme
model_path = "ml_model.pkl"
scaler_path = "scaler.pkl"
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
