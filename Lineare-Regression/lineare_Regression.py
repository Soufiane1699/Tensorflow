import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Erzeugung zufälliger Daten
np.random.seed(0)
X = np.random.randn(100, 1)
y = 3 * X + 2 + 0.5 * np.random.randn(100, 1)

# Tensorflow-Modell für lineare Regression
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])

# Kompilieren des Modells
model.compile(optimizer='sgd', loss='mse')


# Training des Modells
history = model.fit(X, y, epochs=50)

# Vorhersage für neue Daten
X_test = np.array([[1.5], [2.5]])
y_pred = model.predict(X_test)

# Visualisierung der Ergebnisse
plt.scatter(X, y)
plt.plot(X_test, y_pred, color='red')
plt.scatter(X_test, y_pred, color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Lineare Regression')
plt.legend(['Regressionsgerade', 'Vorhersage'])
plt.show()
