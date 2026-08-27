import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

DATASET_PATH = "../data/sudoku_ml_dataset.npz"
MODEL_PATH = "sudoku_ml_model.keras"
EPOCHS = 100 #80binlik sudokuyu 100 kere baştan sona çöz
BATCH_SIZE = 128 #her 128 sudokuda bir dönüp cevap anahtarına bak 

print("DATASET YÜKLENİYOR")
data = np.load(DATASET_PATH)
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]

print(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
print(f"X_test : {X_test.shape} | y_test : {y_test.shape}")

print("MODEL OLUŞTURULUYOR")
# TAM EVRİŞİMLİ SİNİR AĞI (CNN)
model = models.Sequential([ 
    layers.Input(shape=(9, 9, 10)), #9-9-10 katmanın çevirdiğimiz sudokuyu alır
    
    layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'), #modeli 3*3lük arar
    layers.BatchNormalization(), #sayıları dengeler
    
    layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'), #5
    layers.BatchNormalization(),
    
    layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'), #7
    layers.BatchNormalization(),
    
    layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'), #9
    layers.BatchNormalization(),
    
    #Görme alanını (Receptive Field) tüm tahtaya yaymak için
    # 5. Katman
    layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    # 6. Katman
    layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    
    layers.Conv2D(9, kernel_size=(1, 1), padding='same', activation='softmax')
])

model.compile( #model kuralları
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), #Adam tekniği ile 0.0005 öğrenme hızıyla ilerle
    loss="sparse_categorical_crossentropy", #ceza sistemi
    metrics=["accuracy"] #doğruluk puanı
)
model.summary()

callbacks_list = [
    callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1), #modelin rekor kırdığı en iyi hali kaydeder 
    callbacks.EarlyStopping(monitor="val_accuracy", patience=10, mode="max", restore_best_weights=True, verbose=1), #model 5 adımdan fazla aynıysa erken durdurur
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=0.00001, verbose=1) #model tıkandığında öğrenme hızını düşürür
]

print("MODEL EĞİTİMİ BAŞLIYOR")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True,
    callbacks=callbacks_list
)

print("MODEL TEST EDİLİYOR")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1) #eğitim bitti şiimdi hiç görmediklerini çöz
print(f"Test Accuracy : %{test_accuracy * 100:.2f}")

model.save(MODEL_PATH)