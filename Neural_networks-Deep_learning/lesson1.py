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
