import tensorflow as tf
def test_gpu():
    if (tf.test.gpu_device_name()):
        print(f"{tf.test.gpu_device_name()}")

def main():
    test_gpu()


if __name__ == "__main__":
    main()