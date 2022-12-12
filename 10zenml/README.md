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