## Test for distributed training
## https://www.tensorflow.org/guide/distributed_training

import tensorflow as tf

print(tf.version.VERSION)

strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"], cross_device_ops=tf.distribute.NcclAllReduce())
# strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(1,))
    ])
    # it shoud be inside the strategy scope
    model.compile(loss='mae', optimizer='sgd')

dataset = tf.data.Dataset.from_tensors(([1.], [1.]))

# https://stackoverflow.com/questions/72740907/tensorflow-cant-apply-sharing-policy-file-when-using-mirrored-strategy
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE


## you will have map and shuffle 
dataset = dataset.with_options(options)
dataset = dataset.repeat(100).batch(10 * strategy.num_replicas_in_sync)
print(f"num_replics_in_sync: {strategy.num_replicas_in_sync}")
# prefetch take advantage of multi-threading, 1=AUTOTUNE
dataset = dataset.prefetch(tf.data.AUTOTUNE)

model.fit(dataset, epochs = 5)
model.evaluate(dataset)


