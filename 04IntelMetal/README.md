# About this project
Install Metal Plugin for TensorFlow Macos to use GPU with TensorFlow 2.6 on a Intel Macbook 2019 with AMD 555X

## Install Python3.8 VENV
```
brew install python@3.8
brew info python@3.8
```
to see the installation path of `python3.8`
```console
Python has been installed as
  /usr/local/opt/python@3.8/bin/python3
```

Since at the timepoint of writing, only `python3.8` is supported, we need to make a VENV on a specific version
```
/usr/local/opt/python@3.8/bin/python3 -m venv ~/metal3.8
```
The path `~/metal3.8` is where you would like to have your virtual env

## Install the TensorFlow_Maco and Tensorflow Metal
first we need to activate the virtual env, we have created sofar
```
source ~/metal3.8/bin/activte
```

You now see the command prompt with prefix indicating the virtual env is active.
```
(metal3.8) <user>@host % 
```

Update the pip
```
python3 -m pip install -U pip
```

Install the tensorflow-macos version 2.6.0 and tensorflow-metal
```
SYSTEM_VERSION_COMPAT=0 pip install tensorflow-macos==2.5.0 tensorflow-federated==0.19.0 tensorflow-metal 
```

alternatively you can installed the tensorflow-macos==2.6.0 without the tensorflow-federated
```
SYSTEM_VERSION_COMPAT=0 pip install tensorflow-macos==2.6.0 tensorflow-metal
```

we need to use `SYSTEM_VERSION_COMPAT=0`, otherwise an error will occur.
More info regarding the error can be found:
* https://developer.apple.com/forums/thread/683757
* https://developer.apple.com/metal/tensorflow-plugin/

## Examing the GPU support in TensorFlow
If you haven't activated virtual env, you need to do it first
```
source ~/metal3.8/bin/activte
python3
```

In the interactive python3 shell type the following to see the GPU support
```
import tensorflow as tf
print(f"{tf.test.gpu_device_name()}")
```

You shall now see something similar to the following GPU info appears:
```console
2021-11-20 19:06:03.245890: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  SSE4.2 AVX AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
Metal device set to: AMD Radeon Pro 555X

systemMemory: 16.00 GB
maxCacheSize: 2.00 GB

2021-11-20 19:06:03.246695: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:305] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
2021-11-20 19:06:03.247104: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:271] Created TensorFlow device (/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)
/device:GPU:0
```

## Deactivate the virtual env
Should you are still in the VENV, you can deactivate it with
```
deactivate
```

## Uninstall
```
 pip3 uninstall tensorflow-macos==2.5.0 tensorflow-federated==0.19.0 tensorflow-metal

```

## See the pip versions of Federated 
```
pip index versions tensorflow-federated
```



