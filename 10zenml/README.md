# About
@author: Yingding Wang

This repo creates an example mnist zenml pipeline on apple silicon with gpu training on tensorflow-macos using the tensorflow-metal gpu plugin.

## Install packages 
```
python3 -m pip install -r requirements.txt --no-cache
```
## Remove all package from venv
```
python3 -m pip freeze | xargs pip uninstall -y
python3 -m pip list
```

## Start tensorboard
After the training of tensorflow model, log folder will be generated at `<project_root>/logs/fit`

To use a this relative logdir `./logs/fit` with tensorboard
```python
cd <project_root>
# tensorboard --logdir path_to_current_dir
tensorboard --logdir ./logs/fit
```

You will see in the console the following outputs:
```console
tensorboard --logdir ./logs/fit
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.11.0 at http://localhost:6006/ (Press CTRL+C to quit)
```

Open the browser and access tensorboard from the URL as prompt `http://localhost:6006/`

### To see the evaluation of acc vs iteration
1. close the density groups
2. open `evaluation_accuracy_vs_iterations`
3. open `evaluation_loss_vs_iterations`

## Issue
### Zenml tfx dependency
zenml is depending on ml-metadata, and which has no aarch64 package
* https://github.com/google/ml-metadata/issues/143

Solution: zenml 0.30.0 works with `python3.10` and also removed dependencies to `tfx` and `ml-metadata`

### Tensorflow-macos and Tensorflow-metal Optimizer XLA Path issue
* https://developer.apple.com/forums/thread/721619

Workarround:
use the legacy adam optimizer

```python
from tensorflow.keras.optimizers.legacy import Adam
```

instead of
```
from tensorflow.keras.optimizers import Adam
```

## Reference:
* Keras callback: https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
* N epoch callback: https://stackoverflow.com/questions/51025269/creating-a-keras-callback-that-activates-every-n-epochs
* custom callback: https://www.tensorflow.org/guide/keras/custom_callback
* tensorboard: https://stackoverflow.com/questions/42112260/how-do-i-use-the-tensorboard-callback-of-keras