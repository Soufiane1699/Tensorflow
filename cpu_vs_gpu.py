import tensorflow as tf
import time

def matrix_multiplikation(matrix_size, device_name):
    with tf.device(device_name):
        A = tf.random.normal([matrix_size, matrix_size], dtype=tf.float32)
        B = tf.random.normal([matrix_size, matrix_size], dtype=tf.float32)
        start_time = time.time()
        C = tf.matmul(A, B)
        elapsed_time = time.time() - start_time
    return elapsed_time

def main():
    matrix_size = 5000

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
