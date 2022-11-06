'''
This testing module tests some basic function of the venv
'''
import tensorflow as tf
from utils.helper import create_default_tf_checkpoint_subfolder

print(f"Tensorflow verion: {tf.__version__}")

tf_checkpoint_subfolder = create_default_tf_checkpoint_subfolder()
print(f"{tf_checkpoint_subfolder}")

