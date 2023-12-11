### simple MLP classifier
import numpy as np
import tensorflow as tf

# from zenml.steps import step
from zenml import step
from util import get_local_time_str, MultiEpochProgbarLogger

# WARNING tells M1/M2 tf.optimizers.Adam() is slow on M1/M2, legacy Adam is fast

from tensorflow.keras.optimizers.legacy import Adam
# from tensorflow.optimizers import Adam

# Work around for no XLA path support with using Adam form legacy,
# instead of the default tf.optimizers.Adam()
# https://developer.apple.com/forums/thread/721619

@step
def tf_gpu_trainer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
)-> None: 
    """Train a tensorflow classfier."""
    print("tf_gpu_trainer")
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    # print(np.unique(y_train))
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
         tf.keras.layers.Dropout(0.1),
         tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    batch_size = 1200
    epochs = 250
    log_dir = "logs/fit/" + get_local_time_str(target_tz_str='Europe/Berlin')
    # https://www.tensorflow.org/tensorboard/get_started


    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # nbatch_progbar_callback = NBatchProgBarLogger(display_per_batches=10)
    # nbatch_callback = NBatchLogger(display=10)
    # progressbar_callback = tf.keras.callbacks.ProgbarLogger()
    multiEpochProgbarLogger = MultiEpochProgbarLogger(count_mode="steps",display_per_epoch=20)
    

    # 1D Integer encoded target sparse_categorical_crossentropy as loss funciton
    # one-hot encoded with categorical_crossentropy
    # model.compile(optimizer=tf.optimizers.Adam(), loss="categorical_crossentropy", metrics=['accuracy'])
    
    '''sparse categorical crossentropy'''
    model.compile(optimizer=Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    # With tensorflow-metal we need to use the legacy Adam optimizer
    # https://developer.apple.com/forums/thread/721619
    # model.compile(optimizer=Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=1,
        validation_data=(X_test, y_test),
        callbacks=[multiEpochProgbarLogger, tensorboard_callback]
    )
    print(model.summary())

    # https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
    


