'''
This model trains a fashion mnist model, and convert to tensorflow lite mode with the following exposed functions
* train
* infer
* save
* restore
functions
'''
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os

# print(f"Tensorflow verion: {tf.__version__}")
print(f"Tensorflow verion: {tf.version.VERSION}")

# quick version check for this code
assert tf.version.VERSION == "2.10.0"

IMG_SIZE = 28

class Model(tf.Module):

    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(IMG_SIZE, IMG_SIZE), name='flatten'),
            tf.keras.layers.Dense(units=128, activation='relu', name='dense_1'),
            tf.keras.layers.Dense(units=10, activation=None, name='dense_2')
        ])

        self.model.compile(
            optimizer='sgd',
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        )

    # The `train` function takes a batch of input images and labels
    @tf.function(input_signature=[
        tf.TensorSpec([None, IMG_SIZE, IMG_SIZE], tf.float32),
        tf.TensorSpec([None, 10], tf.float32)
    ])
    def train(self, x, y):
        with tf.GradientTape() as tape:
            prediction = self.model(x)
            loss = self.model.loss(y, prediction)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )    
        result = {"loss": loss}
        return result

    @tf.function(input_signature=[
        tf.TensorSpec([None, IMG_SIZE, IMG_SIZE], tf.float32),
    ])
    def infer(self, x):
        logits = self.model(x)
        probabilities = tf.nn.softmax(logits, axis=-1)
        return {
            "output": probabilities,
            "logits": logits
        }

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[], dtype=tf.string)
    ])
    def save(self, checkpoint_path):
        tensor_names = [weight.name for weight in self.model.weights]
        tensors_to_save = [weight.read_value() for weight in self.model.weights]
        tf.raw_ops.Save(
            filename=checkpoint_path, tensor_names=tensor_names,
            data=tensors_to_save, name='save'
        )
        return {
            "checkpoint_path": checkpoint_path
        }
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[], dtype=tf.string)
    ])
    def restore(self, checkpoint_path):
        restored_tensors = {}
        for var in self.model.weights:
            restored = tf.raw_ops.Restore(
                file_pattern=checkpoint_path, tensor_name=var.name,
                dt=var.dtype, name='restore'
            )
            var.assign(restored)
            restored_tensors[var.name] = restored
        return restored_tensors


fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

###
# Preprocess the data
# Pixel values in this dataset are between 0 and 255, and must be normalized to a value
# between 0 and 1 for processing by the model. Divide the values by 255 to make this adustment
###
train_images = (train_images / 255.0).astype(np.float32)
test_images = (test_images / 255.0).astype(np.float32)

## Convert the data labels to categorical values by performing one-hot encoding
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

NUM_EPOCHS = 100 #100
BATCH_SIZE = 100

epochs = np.arange(1, NUM_EPOCHS + 1, 1)
losses = np.zeros([NUM_EPOCHS])
m = Model()

train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_ds = train_ds.batch(BATCH_SIZE)

for i in range(NUM_EPOCHS):
    for x, y in train_ds:
        result = m.train(x, y)

    losses[i] = result['loss'] 
    if (i + 1) % 10 == 0:
        print(f"Finished {i+1} epochs")
        print(f"   loss: {losses[i]:.3f}")

# Save the trained weights to a checkpoint
current_path = os.path.abspath(os.getcwd())
current_model_path = os.path.join(current_path, "tmp")

if not os.path.exists(current_model_path):
    os.makedirs(current_model_path)
print(current_path)
m.save(f"{current_model_path}/model.ckpt")

###
# Plot the training result
###

plt.plot(epochs, losses, label='Prfe-training')
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch')
plt.ylabel('Loss [Cross Entropy]')
plt.legend()

plt.show()