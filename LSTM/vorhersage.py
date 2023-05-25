import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Daten vorbereiten
data = np.array([95,96,97,98,99,100])
data = data.reshape((1, 6, 1))

# Modell erstellen
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(6, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Modell trainieren
model.fit(data, np.array([101]), epochs=450, verbose=0)

# Vorhersage machen
x_input = np.array([96,97,98,99,100,101]).reshape((1, 6, 1))
yhat = model.predict(x_input)
print(yhat)
