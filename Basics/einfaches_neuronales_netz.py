import tensorflow as tf

# Definieren der Eingabedaten
x_train = [1.0, 2.0, 3.0, 4.0]
y_train = [2.0, 4.0, 6.0, 8.0]

# Definieren des Modells
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# Kompilieren des Modells
model.compile(optimizer=tf.optimizers.Adam(0.1), loss='mean_squared_error')

# Trainieren des Modells
model.fit(x_train, y_train, epochs=100)

# Vorhersage
print(model.predict([5.0]))