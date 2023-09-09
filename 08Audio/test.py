import tensorflow as tf
import tensorflow_io as tfio


def test_gpu():
    if (tf.test.gpu_device_name()):
        print(f"{tf.test.gpu_device_name()}")


def test_tfio():
    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"], cross_device_ops=tf.distribute.NcclAllReduce())
    with strategy.scope():
        path = tf.keras.utils.get_file(
            fname='setero.wav', 
            origin='https://upload.wikimedia.org/wikipedia/commons/e/ef/Bach_C_minor_Passacaglia_variation_with_ostinato_in_treble.wav',
            )
        
        print(path)
        # Open the file as a `tf.data.Dataset`
        audio_ds = tfio.IODataset.from_audio(path)
        # => <AudioIODataset shapes: (2,), types: tf.int16>

        # Or as a virtual Tensor that loads what's needed.
        audio_io = tfio.IOTensor.from_audio(path)
        # =>  <AudioIOTensor: spec=TensorSpec(shape=(880640, 2), dtype=tf.int16, name=None), rate=44100>
        slice = audio_io[:1024]


def main():
    test_gpu()
    test_tfio()

if __name__ == "__main__":
    main()
