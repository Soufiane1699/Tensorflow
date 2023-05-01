import tensorflow as tf

def main():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPU(s) gefunden:")
        for gpu in gpus:
            print(f" - {gpu}")

        # GPU-Modell ausgeben
        gpu_model = tf.test.gpu_device_name()
        print(f"\nGPU-Modell: {gpu_model}")
    else:
        print("Keine GPUs gefunden. Stellen Sie sicher, dass Ihre GPU-Treiber und CUDA korrekt installiert sind.")


if __name__ == "__main__":
    main()

# Die korrekte Ausgabe, wenn eine GPU erkannt wurde lautet:
# GPU-Modell: /device:GPU:0
