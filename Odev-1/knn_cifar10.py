import numpy as np
import pickle

# ============================================================
# CIFAR-10 Üzerinde k-NN Sınıflandırma
# Derin Sinir Ağları - Ödev 1
# ============================================================

# CIFAR-10 sınıf isimleri
sinif_isimleri = [
    "Uçak", "Otomobil", "Kuş", "Kedi", "Geyik",
    "Köpek", "Kurbağa", "At", "Gemi", "Kamyon"
]

# ---------- VERİ YÜKLEME ----------
print("CIFAR-10 verisi yükleniyor...")

with open("cifar10_train.pkl", "rb") as f:
    train_verisi = pickle.load(f)

with open("cifar10_test.pkl", "rb") as f:
    test_verisi = pickle.load(f)

x_train = train_verisi["data"]
y_train = train_verisi["labels"]
x_test = test_verisi["data"]
y_test = test_verisi["labels"]

print(f"Eğitim seti: {x_train.shape[0]} görüntü")
print(f"Test seti  : {x_test.shape[0]} görüntü")
print(f"Görüntü boyutu: {x_train.shape[1]}x{x_train.shape[2]}x{x_train.shape[3]}")

# ---------- VERİYİ DÜZLEŞTİRME ----------
# Her görüntüyü tek boyutlu vektöre çevir (32x32x3 = 3072)
x_train_duz = x_train.reshape(x_train.shape[0], -1).astype(np.float64)
x_test_duz = x_test.reshape(x_test.shape[0], -1).astype(np.float64)

# ---------- PERFORMANS İÇİN ALT KÜME SEÇİMİ ----------
# 50.000 eğitim verisi ile k-NN çok yavaş olacağından alt küme kullanıyoruz
EGITIM_SAYISI = 5000
TEST_SAYISI = 500

# Rastgele alt küme seç (tekrarlanabilirlik için seed)
np.random.seed(42)
egitim_indeksler = np.random.choice(x_train_duz.shape[0], EGITIM_SAYISI, replace=False)
test_indeksler = np.random.choice(x_test_duz.shape[0], TEST_SAYISI, replace=False)

x_egitim = x_train_duz[egitim_indeksler]
y_egitim = y_train[egitim_indeksler]
x_test_secili = x_test_duz[test_indeksler]
y_test_secili = y_test[test_indeksler]

print(f"\nKullanılan eğitim verisi: {EGITIM_SAYISI}")
print(f"Kullanılan test verisi  : {TEST_SAYISI}")

# ---------- KULLANICIDAN GİRDİ ALMA ----------
print("\n" + "=" * 50)
print("UZAKLIK METRİĞİ SEÇİMİ")
print("=" * 50)
print("1 - L1 (Manhattan) Uzaklığı")
print("2 - L2 (Öklid) Uzaklığı")

while True:
    secim = input("\nSeçiminiz (1 veya 2): ").strip()
    if secim in ("1", "2"):
        break
    print("Geçersiz seçim! Lütfen 1 veya 2 giriniz.")

if secim == "1":
    metrik = "L1"
    print(">>> L1 (Manhattan) uzaklığı seçildi.")
else:
    metrik = "L2"
    print(">>> L2 (Öklid) uzaklığı seçildi.")

print("\n" + "=" * 50)
print("K DEĞER SEÇİMİ")
print("=" * 50)

while True:
    k_girdi = input("k değerini giriniz (örn: 3, 5, 7): ").strip()
    if k_girdi.isdigit() and int(k_girdi) > 0:
        k = int(k_girdi)
        break
    print("Geçersiz giriş! Lütfen pozitif bir tam sayı giriniz.")

print(f">>> k = {k} seçildi.")

# ---------- k-NN SINIFLANDIRMA ----------
print("\n" + "=" * 50)
print(f"k-NN Sınıflandırma Başlıyor (k={k}, metrik={metrik})")
print("=" * 50)

dogru_sayisi = 0
toplam = TEST_SAYISI

for i in range(toplam):
    # Mevcut test görüntüsü
    test_noktasi = x_test_secili[i]

    # Tüm eğitim verilerine olan uzaklığı hesapla
    if metrik == "L1":
        # Manhattan uzaklığı: |x1 - x2| toplamı
        uzakliklar = np.sum(np.abs(x_egitim - test_noktasi), axis=1)
    else:
        # Öklid uzaklığı: sqrt((x1 - x2)^2 toplamı)
        uzakliklar = np.sqrt(np.sum((x_egitim - test_noktasi) ** 2, axis=1))

    # En yakın k komşuyu bul
    en_yakin_indeksler = np.argsort(uzakliklar)[:k]
    en_yakin_etiketler = y_egitim[en_yakin_indeksler]

    # Çoğunluk oylaması ile sınıf belirle
    sinif_sayilari = np.bincount(en_yakin_etiketler, minlength=10)
    tahmin = np.argmax(sinif_sayilari)

    # Doğruluğu kontrol et
    if tahmin == y_test_secili[i]:
        dogru_sayisi += 1

    # İlerleme göstergesi
    if (i + 1) % 50 == 0 or i == 0:
        print(f"  İşlenen: {i + 1}/{toplam}  |  Anlık doğruluk: {dogru_sayisi / (i + 1) * 100:.2f}%")

# ---------- SONUÇLAR ----------
dogruluk = dogru_sayisi / toplam * 100

print("\n" + "=" * 50)
print("SONUÇLAR")
print("=" * 50)
print(f"Uzaklık metriği : {metrik} ({'Manhattan' if metrik == 'L1' else 'Öklid'})")
print(f"k değeri         : {k}")
print(f"Eğitim verisi    : {EGITIM_SAYISI}")
print(f"Test verisi      : {TEST_SAYISI}")
print(f"Doğru tahmin     : {dogru_sayisi}/{toplam}")
print(f"Doğruluk oranı   : {dogruluk:.2f}%")
print("=" * 50)
