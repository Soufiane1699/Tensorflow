import tensorflow as tf
import time

def main():
    # sicherstellen, dass eine GPU erkannt wird
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("Keine GPUs gefunden. Stellen Sie sicher, dass Ihre GPU-Treiber richtig installiert sind.")
        return

    # Erstellen von zwei zufälligen Matrizen
    matrix_size = 1000
    A = tf.random.normal([matrix_size, matrix_size], dtype=tf.float32)
    B = tf.random.normal([matrix_size, matrix_size], dtype=tf.float32)

    # Matrixmultiplikation mit TensorFlow
    start_time = time.time()
    C = tf.matmul(A, B)
    elapsed_time = time.time() - start_time

    print(f"Matrixmuliplikation abgeschlossen in {elapsed_time:.4f} Sekunden.")

if __name__ == "__main__":
    main()

# Beispielausgabe von meiner Nvidia 3070
# Matrixmuliplikation abgeschlossen in 0.8190 Sekunden.