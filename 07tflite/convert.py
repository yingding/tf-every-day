import os
import tensorflow as tf
from custom_model import Model, IMG_SIZE

# Save the trained weights to a checkpoint
current_path = os.path.abspath(os.getcwd())
current_model_path = os.path.join(current_path, "tmp")

if not os.path.exists(current_model_path):
    os.makedirs(current_model_path)
print(current_path)

m = Model()

m.restore(f"{current_model_path}/model.ckpt")

m.model.summary()
