import tensorflow as tf
import tensorflow_io as tfio
path = tf.keras.utils.get_file('setero.wav', 'https://upload.wikimedia.org/wikipedia/commons/e/ef/Bach_C_minor_Passacaglia_variation_with_ostinato_in_treble.wav')

# Open the file as a `tf.data.Dataset`
audio_ds = tfio.IODataset.from_audio(path)
# => <AudioIODataset shapes: (2,), types: tf.int16>

# Or as a virtual Tensor that loads what's needed.
audio_io = tfio.IOTensor.from_audio(path)
# =>  <AudioIOTensor: spec=TensorSpec(shape=(880640, 2), dtype=tf.int16, name=None), rate=44100>
slice = audio_io[:1024]