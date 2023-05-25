import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Zufällige Trainingsdaten erstellen
np.random.seed(0)
x_train = np.random.random((1000, 10))
y_train = 10 * np.sum(x_train, axis=1) + np.random.randn(1000) * 0.33

# Einfaches Modell erstellen
model = keras.Sequential([
    layers.Dense(1, input_dim=10, kernel_initializer='normal', activation='linear'),
])

# Modell kompilieren
model.compile(loss='mean_squared_error', optimizer='adam')

# Modell trainieren
model.fit(x_train, y_train, epochs=50, batch_size=10)

# Testdaten vorbereiten
x_test = np.random.random((10, 10))

# Vorhersagen machen
predictions = model.predict(x_test)
print(predictions)
