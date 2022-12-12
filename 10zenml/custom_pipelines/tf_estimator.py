### svc support vector classifier
import numpy as np
import tensorflow as tf

from zenml.steps import step

@step
def tf_gpu_trainer(
    X_train: np.ndarray,
    y_train: np.ndarray,
)-> None: 
    """Train a tensorflow classfier."""
    print("tf_gpu_trainer")
    print(X_train.shape)
    print(y_train.shape)
    if (tf.test.gpu_device_name()):
        print(f"{tf.test.gpu_device_name()}")

    model = tf.keras.Sequential([
         tf.keras.layers.Flatten(input_shape=(64,)),
         tf.keras.layers.Dense(16, activation='relu'),
         tf.keras.layers.Dense(10)
    ])
    
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    print(model.summary())


