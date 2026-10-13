import numpy as np
import matplotlib.pyplot as plt

# TensorFlow / Keras importok
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Reprodukálhatóság biztosítása (random seed rögzítése)
tf.random.set_seed(42)

print(f"TensorFlow verzió: {tf.__version__}")

# Adatok betöltése
(x_train_full, y_train_full), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Normalizálás: A pixelértékeket [0, 255] tartományból képezzük le a [0.0, 1.0] intervallumra.
# Miért fontos ez? A gradiens alapú optimálók lényegesen stabilabban konvergálnak kis skálájú bemenetek esetén.
x_train_full = x_train_full / 255.0
x_test = x_test / 255.0

# Tanító és validációs halmaz szétválasztása
x_train, x_valid = x_train_full[:-5000], x_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Kirajzolunk 16 mintaképet a tanítóhalmazból
plt.figure(figsize=(8, 8))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    # A képek 28x28-as mátrixok, szürkeárnyalatos megjelenítéssel
    # A plt.imshow() segítségével a 2D tömbünket képpé alakítjuk.
    # A cmap='gray' azért szükséges, mert a képek 1 csatornás szürkeárnyalatosak, nem RGB színes képek.
    plt.imshow(x_train[i], cmap='gray')
    plt.title(class_names[y_train[i]])
    plt.axis('off')

plt.suptitle("Fashion-MNIST Mintaképek", fontsize=16)
plt.tight_layout()
plt.show()

# A Sequential modell egy lineáris réteg-köteg (stack).
# A bemenettől a kimenetig az adatok sorban haladnak át a rétegeken.
model_tf = keras.Sequential([

    # 1. FLATTEN RÉTEG:
    # A képeink 2D-s mátrixok (28x28 pixel). A sűrűn összefüggő (Dense) rétegek
    # viszont 1D-s vektort várnak bemenetként. A Flatten átalakítja a (28, 28)
    # alakú tömböt egy (784,) elemű 1D vektorrá.
    # FIGYELEM: Ez nem végez súlyozott összeadást, nincs tanulható paramétere!
    layers.Flatten(input_shape=[28, 28]),

    # 2. REJTETT RÉTEG 1 (Dense):
    # 'Dense' = Sűrűn kötött / Fully Connected réteg. Minden bemeneti neuron össze van kötve
    # a réteg minden egyes neuronjával.
    # Neuronok száma: 128. (KÍSÉRLETEZÉSI PONT: Próbáljátok ki 64-gyel vagy 256-tal!)
    # Aktivációs funkció: 'relu' (Rectified Linear Unit, f(x) = max(0, x)).
    # Ez hozza be a nemlinearitást, ami nélkül a hálózat csak egy komplex lineáris regresszió lenne.
    layers.Dense(128, activation="relu"),

    # 3. REJTETT RÉTEG 2 (Dense):
    layers.Dense(64, activation="relu"),

    # 4. KIMENETI RÉTEG (Dense):
    # A neuronok száma PONTOSAN 10 kell legyen, mert 10 osztályunk van (0-9).
    # Aktiváció: 'softmax'. A softmax a kimeneti értékeket egy olyan valószínűségi
    # eloszlásba transzformálja, ahol az értékek összege 1.0 (100%).
    layers.Dense(10, activation="softmax")
])

# A modell struktúrájának és paraméterszámának ellenőrzése
model_tf.summary()

model_tf.compile(
    # Optimáló eljárás: Adam (Adaptive Moment Estimation)
    # A sztochasztikus gradiens leereszkedés (SGD) egy fejlett, adaptív tanulási rátát használó változata.
    optimizer="adam",

    # Veszteségfüggvény (Loss function):
    # 'sparse_categorical_crossentropy'-t használunk, mert az y_train címkéink egész számok (0..9).
    # Ha one-hot encoded kódolásúak lennének a címkék, a 'categorical_crossentropy'-t kellene használni.
    loss="sparse_categorical_crossentropy",

    # Metrika, amit nyomon követünk a tanítás alatt (a veszteségértéken kívül)
    metrics=["accuracy"]
)

# Tanítás
history = model_tf.fit(
    x_train, y_train,
    epochs=10,  # KÍSÉRLETEZÉSI PONT: Hány epoch-ig tanítsunk? Nézzétek meg, mikor lép fel túltanulás (overfitting)!
    batch_size=32, # Batch méret: Hány mintánként frissítse a súlyokat az optimáló.
    validation_data=(x_valid, y_valid)
)
# Keras history kirajzolása
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 1. Veszteség (Loss) grafikon
ax1.plot(history.history['loss'], label='Tanító Loss', color='blue')
ax1.plot(history.history['val_loss'], label='Validációs Loss', color='red', linestyle='--')
ax1.set_title('Keras: Tanítási és Validációs Veszteség')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

# 2. Pontosság (Accuracy) grafikon
ax2.plot(history.history['accuracy'], label='Tanító Accuracy', color='blue')
ax2.plot(history.history['val_accuracy'], label='Validációs Accuracy', color='red', linestyle='--')
ax2.set_title('Keras: Tanítási és Validációs Pontosság')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True)

plt.show()
