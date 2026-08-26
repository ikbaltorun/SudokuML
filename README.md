# 🧩 CNN Destekli Hibrit Sudoku Çözücü
<p align="center">
  <img src="sudoku2.gif" width="250">
</p>
Bu proje; derin öğrenme ile katı mantıksal kuralları birleştiren, yüksek performanslı hibrit bir Sudoku çözücüdür. Geleneksel arama algoritmaları karmaşık bulmacalarda arama uzayı patlamasıyla (search space explosion) karşılaşırken, saf yapay zeka modelleri temel oyun kurallarını ihlal edebilir. Bu proje, hamle tahmini için Evrişimli Sinir Ağlarını (CNN) ve kesin doğruluk garantisi için Top-K Güvenilirlik Backtracking (Geri İzleme) algoritmasını bir araya getirerek bu ikilemi çözer.

---

## 🚀 Öne Çıkan Özellikler

* **Gelişmiş Veri Boru Hattı ve One-Hot Encoding:** Tahta durumlarını kategorik çok kanallı tensörlere (`9x9x10`) dönüştürerek sayısal yanlılığı ortadan kaldırır (modelin 8 sayısını 4'ten "matematiksel olarak büyük" sanmasını engeller).
* **Derin Öğrenme Mimarisi (CNN):** Satırlar, sütunlar ve 3x3'lük alt ızgaralar arasındaki küresel tahta geometrisini yakalamak için uzaysal dolgu (`same`) ve toplu normalleştirme (`batch normalization`) içeren çok katmanlı `Conv2D` filtreleri kullanır.
* **Top-K Sezgisel Backtracking:** Körlemesine tahmin yapmak veya rastgele aramak yerine, çözücü CNN'in olasılık dağılımını sorgular, aday hamleleri güven skoruna göre sıralar ve çıkmaz sokaklarda zarifçe geri izleme yapar.
* **Hibrit %100 Otomatik Test Başarısı:** Modelin tek hamlelik ham tahmin başarısı optimize edilmişken, arkadaki hibrit backtracking mantığı sayesinde standart ve ekstrem veri setlerinde **%100 kusursuz çözüm garantisi** sunar.
* **Etkileşimli Web Arayüzü:** Gerçek zamanlı görselleştirme için Streamlit ile geliştirilmiş, tam duyarlı (responsive) modern bir karanlık mod (dark-mode) arayüzü içerir.

---

## 🏗️ Proje Mimarisi
```text
SudokuML/
│
├── data/
│   ├── generate_dataset.py    # 100.000+ sentetik Sudoku bulmacası ve çözümü üretir
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
3. **İnteraktif web uygulamasını çalıştırın:**
   ```bash
   streamlit run app.py
   ```
4. **Otomatik kıyaslama ve test paketini çalıştırın (100 Bulmaca):**
   ```bash
   cd solver
   python test_solver.py
   ```
5. **Ekstrem stres testini çalıştırın (AI Escargot ve 17-İpuculu Minimal):**
    ```bash
    cd solver
    python extreme_test.py
    ```

---

## 📈 Geliştirme ve Optimizasyon Süreci
Bu projeyi geliştirirken modelin başlarda düşük doğruluk oranlarıyla takılması ve bunu aşamalı olarak nasıl çözdüğümüz projenin en değerli aşaması oldu:

1. **İlk Durum ve Düşük Başarı (%65):**

İlk denemelerde daha küçük veri seti ve varsayılan ayarlarla başlandığında modelin doğruluğu %65 civarında takılıyor ve erken kesiliyordu.
Çözüm / Müdahale: Veri seti boyutu 100.000+ bulmacaya çıkarıldı. EarlyStopping sabrı (patience) artırıldı, ReduceLROnPlateau ile öğrenme hızı dinamik olarak düşürüldü ve batch_size 32 olarak optimize edildi.

2. **Optimizasyon Sonrası Sınav Başarısı (%68.45):**

Yapılan iyileştirmelerle birlikte model EarlyStopping ile 29. turda en iyi ağırlıklarına (%19. epoch) ulaştı ve gerçek sınav başarısı (val_accuracy) %68.45 olarak tescillendi.
loss ve val_loss değerlerinin birbirine çok yakın olması, modelin ezber yapmadığını (overfitting olmadığını) net bir şekilde kanıtladı.

3. **Hibrit Mimari ile Kesin Çözüm (%100 Garanti):**

Model tek başına her zaman %100 bilmese de, arkada çalışan Top-K Sezgisel Backtracking motoru sayesinde en yüksek olasılıklı hamleler akıllıca denenir. Bu hibrit yapı sayesinde model, 100 zorlu bulmacanın tamamını sıfır hatayla çözer.

## 📊 Canlı Test ve Performans Çıktıları

Projenin toplu benchmark testlerinde (test_solver.py) elde ettiği gerçek çalışma karnesi:

**Toplam Test Edilen Bulmaca:** 100 Adet (Zorluk seviyesi: 45 boş hücre)

**Başarılı Çözüm:** 100

**Başarısız Çözüm:** 0

**Gerçek Sınav Başarısı (Test Accuracy):** %68.45 (Modelin tek hamlelik ham tahmin başarısı)

**Hibrit Sistem Kesin Başarısı:** %100.00 (Backtracking entegrasyonu ile sıfır hata)

**Ortalama Çözüm Adımı:** ~100 adım / bulmaca

**Toplam Test Süresi:** 8 dakika 15 saniye (Bulmaca başına ~4.9 saniye)

---

👩‍💻 Geliştirici
*İkbal Torun*
  
