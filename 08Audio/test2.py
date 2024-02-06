import tensorflow as tf
import tensorflow_io as tfio

def test_gpu():
    if (tf.test.gpu_device_name()):
        print(f"{tf.test.gpu_device_name()}")

def test_tfio(use_gpu: bool = True):
    if use_gpu:
        devices = ["/gpu:0"]
    else:
        devices = ["/cpu:0"]
    
    strategy = tf.distribute.MirroredStrategy(devices=devices, cross_device_ops=tf.distribute.NcclAllReduce())
    with strategy.scope():
        # Read the MNIST data into the IODataset.
        dataset_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
        d_train = tfio.IODataset.from_mnist(
            dataset_url + "train-images-idx3-ubyte.gz",
            dataset_url + "train-labels-idx1-ubyte.gz",
        )

        # Shuffle the elements of the dataset.
        d_train = d_train.shuffle(buffer_size=1024)

        # By default image data is uint8, so convert to float32 using map().
        d_train = d_train.map(lambda x, y: (tf.image.convert_image_dtype(x, tf.float32), y))

        # prepare batches the data just like any other tf.data.Dataset
        d_train = d_train.batch(32)

        # Build the model.
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(512, activation=tf.nn.relu),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(10, activation=tf.nn.softmax),
            ]
        )

        # Compile the model.
        model.compile(
            optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )

        # Fit the model.
        model.fit(d_train, epochs=5, steps_per_epoch=200)


def main():
    test_gpu()
    test_tfio(use_gpu=True)

if __name__ == "__main__":
    main()