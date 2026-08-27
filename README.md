# 🧩 CNN Destekli Hibrit Sudoku Çözücü
<p align="center">
  <img src="sudoku2.gif" width="250">
</p>
Bu proje; derin öğrenme ile katı mantıksal kuralları birleştiren, yüksek performanslı hibrit bir Sudoku çözücüdür. Geleneksel arama algoritmaları karmaşık bulmacalarda arama uzayı patlamasıyla (search space explosion) karşılaşırken, saf yapay zeka modelleri temel oyun kurallarını ihlal edebilir. Bu proje, hamle tahmini için Evrişimli Sinir Ağlarını (CNN) ve kesin doğruluk garantisi için Top-K Güvenilirlik Backtracking (Geri İzleme) algoritmasını bir araya getirerek bu ikilemi çözer.

---

## 🚀 Öne Çıkan Özellikler

- 🧠 **Gelişmiş Veri Boru Hattı ve One-Hot Encoding:** Tahta durumlarını kategorik çok kanallı tensörlere (9x9x10) dönüştürerek sayısal yanlılığı ortadan kaldırır (modelin 8 sayısını 4'ten matematiksel olarak büyük sanmasını engeller).
- 🏗️ **Derin Öğrenme Mimarisi (CNN):** Satırlar, sütunlar ve 3x3'lük alt ızgaralar arasındaki küresel tahta geometrisini yakalamak için uzaysal dolgu (*same*) ve toplu normalleştirme (*batch normalization*) içeren çok katmanlı Conv2D filtreleri kullanır.
- ⚡ **Top-K Sezgisel Backtracking:** Körlemesine tahmin yapmak veya rastgele aramak yerine, çözücü CNN'in olasılık dağılımını sorgular, aday hamleleri güven skoruna göre sıralar ve çıkmaz sokaklarda zarifçe geri izleme yapar.
- 🛡️ **Hibrit %100 Otomatik Test Başarısı:** Modelin tek hamlelik ham tahmin başarısı optimize edilmişken, arkadaki hibrit backtracking mantığı sayesinde standart ve ekstrem veri setlerinde %100 kusursuz çözüm garantisi sunar.
- 🎨 **Etkileşimli Web Arayüzü:** Gerçek zamanlı görselleştirme için Streamlit ile geliştirilmiş, tam duyarlı (responsive) modern bir karanlık mod (dark-mode) arayüzü içerir.

---

## 🏗️ Proje Mimarisi
```text
SudokuML/
│
├── data/
│   ├── generate_dataset.py    # 200.000+ sentetik Sudoku bulmacası ve çözümü üretir
│   └── sudoku_ml_dataset.npz  # Sıkıştırılmış NumPy veri seti (Eğitim/Test ayrımı)
│
├── model/
│   ├── train.py               # CNN mimari tanımı, EarlyStopping ve eğitim döngüsü
│   └── sudoku_ml_model.keras  # Eğitilmiş Keras model ağırlıkları
│
├── solver/
│   ├── ml_solver.py           # Detaylı adım günlükleriyle tekli bulmaca çalıştırıcı
│   ├── test_solver.py         # Otomatik toplu test paketi (100 bulmaca + performans ölçümü)
│   └── extreme_test.py        # Dünyaca ünlü ekstrem bulmacalara karşı stres testi (AI Escargot)
│
├── app.py                     # Etkileşimli Streamlit web uygulaması
└── requirements.txt           # Proje bağımlılıkları
```
---

## 🛠️ Kullanılan Teknolojiler ve Kütüphaneler
* **Python:** Ana programlama dili.
* **TensorFlow / Keras:** Derin öğrenme, `Conv2D` katmanları ve CNN model eğitimi.
* **NumPy:** Tensör manipülasyonları, `npz` veri seti yapılandırması ve matris işlemleri.
* **Streamlit:** İnteraktif web arayüzü framework'ü.

---

## ⚙️ Kurulum ve Çalıştırma

1. **Repoyu klonlayın:**
   ```bash
   git clone https://github.com/ikbaltorun/SudokuML.git
   cd SudokuML
   ```
2. **Bağımlılıkları yükleyin:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Veri setini oluşturun:**
   ```bash
   cd data
   python generate_dataset.py
   cd ..
   ```
4. **Modeli eğitin:**
   ```bash
   cd model
   python train.py
   cd ..
   ```
5. **İnteraktif web uygulamasını çalıştırın:**
    ```bash
    streamlit run app.py
    ```
6. **Tekli bulmaca çözücüyü çalıştırın (Adım adım log takibi ile):**
   ```bash
   cd solver
   python ml_solver.py
   ```
7. **Otomatik kıyaslama ve test paketini çalıştırın (100 Bulmaca):**
   ```bash
   python test_solver.py
   ```
8. **Ekstrem stres testini çalıştırın (AI Escargot ve 17-İpuculu Minimal):**
   ```bash
   python extreme_test.py
   ```

---

## 📈 İteratif Geliştirme ve Optimizasyon Süreci

Bu projenin en değerli mühendislik aşaması, modelin başlarda yaşadığı tıkanıklıkları teşhis edip, 3 farklı fazda uyguladığımız mimari iyileştirmelerdir:

### 🔴 Faz 1: İlk Prototip ve Darboğaz (< %65 Başarı)
*   **Durum:** İlk denemelerde sığ bir ağ mimarisi (az katmanlı), küçük bir veri seti ve varsayılan hiperparametreler kullanıldı.
*   **Sorun:** Modelin doğruluğu %65'in altında kalarak çok erken tıkanıyordu. Yapay zekanın hata payı yüksek olduğu için, arkada çalışan Backtracking algoritmasına devasa bir arama uzayı yükü kalıyor ve test süreleri çok uzuyordu.

### 🟡 Faz 2: Veri Boru Hattı İyileştirmesi (%68.45 Başarı)
*   **Müdahale:** Modelin ezber (overfitting) yapmasını engellemek için veri seti **100.000+** bulmacaya çıkarıldı. Eğitimi stabilize etmek adına `Batch Size 32` olarak ayarlandı ve `ReduceLROnPlateau` (öğrenme hızını dinamik düşürme) ile `EarlyStopping` eklendi.
*   **Sonuç:** Model 29. turda altın ağırlıklarına ulaşarak gerçek sınav başarısını (val_accuracy) **%68.45'e** çıkardı. Modelin bu yardımı sayesinde algoritma, 100 bulmacalık stres testini bulmaca başına ortalama 100 adımda ve toplam **8 dakika 15 saniyede** tamamladı.

### 🟢 Faz 3: Mimari Derinlik ve Tam Optimizasyon (%76.25 Başarı)🚀
*   **Müdahale:** Modelin hala tahtanın bütününü algılayamadığı (Receptive Field problemi) teşhis edildi. Gösterim alanını tüm tahtaya yaymak için ağ derinleştirilerek **CNN katman sayısı 4'ten 6'ya çıkarıldı**. Artan kapasiteyi beslemek için veri seti **200.000**'e yükseltildi ve eğitim süresi darboğazını aşmak için `Batch Size 128` yapıldı.
*   **Sonuç:** Yapay zeka 28. epoch'ta **%76.25** ile rekor doğruluk oranına ulaştı. Tahmin gücündeki bu büyük sıçrama, algoritmanın deneme-yanılma yükünü inanılmaz derecede hafifletti.

## 📊 Canlı Test ve Performans Çıktıları

| Performans Metriği | Faz 1 (Prototip) | Faz 2 (İlk İyileştirme) | Faz 3 (Son Mimari) 🚀 |
| :--- | :--- | :--- | :--- |
| **Gerçek Sınav Başarısı (CNN)** | < %65.00 | %68.45 | **%76.25** |
| **Hibrit Kesin Başarı (CNN + BT)** | %100.00 | %100.00 | **%100.00** |
| **Ortalama Çözüm Adımı** | Çok Yüksek | ~100 adım / bulmaca | **~57.2 adım / bulmaca** 📉 |
| **Toplam Test Süresi (100 Adet)** | Ölçülmedi | 8 dakika 15 saniye | **4 dakika 26 saniye** ⚡ |
| **Ortalama Çözüm Süresi** | - | ~4.9 saniye / bulmaca | **~2.6 saniye / bulmaca** ⚡ |

---

👩‍💻 Geliştirici
*İkbal Torun*
  
