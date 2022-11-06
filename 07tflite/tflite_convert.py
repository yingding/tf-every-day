import os
import tensorflow as tf
from utils.custom_model import Model, IMG_SIZE
from utils.helper import (
    create_default_tf_checkpoint_subfolder, 
    create_default_tf_savedmodle_subfolder, 
    create_default_tflite_model_path
)

# Save the trained weights to a checkpoint

current_model_path = create_default_tf_checkpoint_subfolder()
print(current_model_path)
# Load the prevous saved checkpoint
m = Model()
m.restore(f"{current_model_path}/model.ckpt")

# show the model summary() 
m.model.summary()

###
# Convert model to TensorFlow Lite format
# https://www.tensorflow.org/lite/examples/on_device_training/overview#convert_model_to_tensorflow_lite_format
###

SAVED_MODEL_DIR = create_default_tf_savedmodle_subfolder()

print(f"{SAVED_MODEL_DIR}")
if SAVED_MODEL_DIR is not None: 
    tf.saved_model.save(
        obj=m,
        export_dir=SAVED_MODEL_DIR,
        signatures={
            'train': m.train.get_concrete_function(),
            'infer': m.infer.get_concrete_function(),
            'save': m.save.get_concrete_function(),
            'restore': m.restore.get_concrete_function(),
        }
    )

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops
    tf.lite.OpsSet.SELECT_TF_OPS # enbalbe TensorFlow ops
]
converter.experimental_enable_resource_variables = True
# converted to a tflite model binary
tflite_model = converter.convert()

tf_lite_model_path = create_default_tflite_model_path('model.tflite')
print(tf_lite_model_path)

# save the converted model
# https://www.tensorflow.org/lite/models/convert/convert_models
with open(tf_lite_model_path, 'wb') as f:
    f.write(tflite_model)