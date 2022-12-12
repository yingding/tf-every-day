### simple MLP classifier
import numpy as np
import tensorflow as tf

from zenml.steps import step
from tensorflow.keras.optimizers.legacy import Adam

@step
def tf_gpu_trainer(
    X_train: np.ndarray,
    y_train: np.ndarray,
)-> None: 
    """Train a tensorflow classfier."""
    print("tf_gpu_trainer")
    print(X_train.shape)
    print(y_train.shape)
    print(np.unique(y_train))
    if (tf.test.gpu_device_name()):
        print(f"{tf.test.gpu_device_name()}")

    # convert to dummy logits with one hot
    # y_logits = tf.one_hot(y_train, depth=10)
    # print(y_logits[0])
    # print(y_logits.shape)

    # https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-one-hot-encoded-array-in-numpy/37323404#37323404
    # n_values = np.max(y_train) + 1
    # y_logits = np.eye(n_values)[y_train]
    # print(y_logits[0])
    # print(y_logits.shape)

    model = tf.keras.Sequential([
         tf.keras.layers.Flatten(input_shape=(64,)),
         tf.keras.layers.Dense(16, activation=tf.nn.relu),
         tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    batch_size = 1200
    epochs = 200

    # 1D Integer encoded target sparse_categorical_crossentropy as loss funciton
    # one-hot encoded with categorical_crossentropy

    # model.compile(optimizer=tf.optimizers.Adam(), loss="categorical_crossentropy", metrics=['accuracy'])

    # With tensorflow-metal we need to use the legacy Adam optimizer
    # https://developer.apple.com/forums/thread/721619
    model.compile(optimizer=Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
    print(model.summary())
    


