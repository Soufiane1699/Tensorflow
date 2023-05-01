import tensorflow as tf
import time

def matrix_multiplikation(matrix_size, device_name):
    with tf.device(device_name):
        # erstellt eine Matrix der Größe matrix_size * matrix_size und Datentyp tf.float32
        A = tf.random.normal([matrix_size, matrix_size], dtype=tf.float32)
        B = tf.random.normal([matrix_size, matrix_size], dtype=tf.float32)
        # Speichert die aktuelle Zeit in der Variablen start_time als Startzeitpunkt
        start_time = time.time()
        # führt eine Matrixmultiplikation zwischen der Matrix A und B durch
        C = tf.matmul(A, B)
        elapsed_time = time.time() - start_time
    return elapsed_time

def main():
    matrix_size = 20000

    # "/cpu:0" definiert das Gerät auf dem die jeweilige Berechnung durchgeführt werden soll
    # TensorFlow Syntax /gerätetyp:geräteindex und das ganze als Zeichenkette
    cpu_time = matrix_multiplikation(matrix_size, "/cpu:0")
    print(f"Matrixmultiplikation auf der CPU abgeschlosse in {cpu_time:.4f} Sekunden.")

    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("Keine GPUs gefunden. Stellen Sie sicher, dass Ihre GPU-Treiber installiert sind.")
        return

    gpu_time = matrix_multiplikation(matrix_size, "/gpu:0")
    print(f"Matrixmultplikation auf der GPU abgeschlossen in {gpu_time:.4f} Sekunden")

    speedup = cpu_time / gpu_time
    print(f"Die GPU war {speedup:.2f}x schneller als die CPU.")

if __name__ == "__main__":
    main()

'''
Matrixmultiplikation auf der CPU abgeschlosse in 43.7825 Sekunden.
2023-05-01 20:11:31.457177: I tensorflow/compiler/xla/stream_executor/cuda/cuda_blas.cc:637] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
Matrixmultplikation auf der GPU abgeschlossen in 0.7519 Sekunden
Die GPU war 58.23x schneller als die CPU.
'''