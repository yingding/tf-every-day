# About this project
Install Metal Plugin for TensorFlow Macos to use GPU with TensorFlow 2.9.2 on:
* M1 Max Macbook pro 2021
* Intel Macbook pro 2019 with AMD 555X

# 1. Install on M1 apple silicon

## Create a VENV
Create a `tff3.9` python venv 
```console
/opt/homebrew/bin/python3.9 -m venv ~/VENV/tff3.9
source ~/VENV/tff3.9/bin/activate
python3 -m pip install --upgrade pip
```


## Install tensorflow macos 2.9.1 and metal 0.5.1, federated 0.34.0

Run on Macosx 12.6.1 M1 apple silicon within a python3.9 venv.
The steps shall not be changed in orders.
<!-- python3 -m pip uninstall -y tensorflow-macos tensorflow-metal tensorflow-federated attrs dm-tree farmhashpy portpicker semantic-version tensorflow-model-optimization tensorflow-privacy tqdm cachetools grpcio dp-accounting numpy matplotlib pandas scikit-learn tensorflow-datasets tensorflow-probability jax jaxlib scipy  -->

```console
# uninstall all packages 
python3 -m pip freeze | xargs pip uninstall -y

python3 -m pip install tensorflow-macos==2.9.2 --no-cache-dir
python3 -m pip install attrs==21.4.0 dp-accounting==0.1.2 matplotlib==3.6.2 pandas==1.5.1 scikit-learn==1.1.3 tensorflow-datasets==4.5.2 tensorflow-probability==0.15 --no-cache-dir

python3 -m pip install farmhashpy==0.4.0 portpicker==1.5 semantic-version==2.6 tensorflow-model-optimization==0.7.3 tensorflow-privacy==0.8.4 --no-dependencies --no-cache-dir

# can not use tensorflow-metal==0.6.0 got a bus error
python3 -m pip install jax==0.3.24 jaxlib==0.3.24 tensorflow-metal==0.5.1 numpy==1.23.4 --no-cache-dir
python3 -m pip install tensorflow-federated==0.34.0 --no-deps --no-cache-dir
```
Reference:
* Uninstall all pip venv packages: https://stackoverflow.com/questions/11248073/what-is-the-easiest-way-to-remove-all-packages-installed-by-pip/11250821#11250821

<!-- # python3 -m pip install attrs==21.4 dp-accounting==0.1.2 matplotlib==3.3.4 pandas==1.1.4 scikit-learn==1.0.2 tensorflow-datasets==4.5.2 tensorflow-probability==0.15 --no-cache-dir -->

# 2. Install on Intel 
## Install Python3.8 VENV
```
brew search python

brew install python@3.8
brew info python@3.8
```
to see the installation path of `python3.8`
```console
Python has been installed as
  /usr/local/opt/python@3.8/bin/python3
```

on the console of M1 apple silicon
```console
Python has been installed as
  /opt/homebrew/bin/python3.8
```

Since at the timepoint of writing, only `python3.8` is supported, we need to make a VENV on a specific version
with Intel chip
```
/usr/local/opt/python@3.8/bin/python3 -m venv ~/metal3.8
```
with Apple silicon
```
/opt/homebrew/bin/python3.8 -m venv ~/metal3.8
```

The path `~/metal3.8` is where you would like to have your virtual env

## (optional) upgrade pip
```console
python3 -m pip install --upgrade pip
```

## (optional) uninstall the previous Tensorflow _Macos and Tensorflow Metal
```console
# uninstall existing tensorflow-macos and tensorflow-metal
python -m pip uninstall -y tensorflow-macos;
python -m pip uninstall -y tensorflow-metal;
python -m pip uninstall -y tensorflow-federated
```

or single line
```
python -m pip uninstall -y tensorflow-macos tensorflow-metal tensorflow-federated
```

## Install the TensorFlow Macos and Tensorflow Metal
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

Search Pypi.org, since pip search is deactivated indefinitively.
```
# browse URL to get the latest version
https://pypi.org/project/tensorflow-federated/
```

#### Install tensorflow macos 2.8.0
You may want to updated pip firt
```console
python3 -m pip install --upgrade pip
```

Run on Macosx 12.3
```console
python3 -m pip install tensorflow-macos==2.8.0 tensorflow-metal==0.4.0;
python3 -m pip install tensorflow-federated==0.20.0
```

Run on Macosx 12.2.1
```console
SYSTEM_VERSION_COMPAT=0 pip install tensorflow-macos==2.8.0 tensorflow-metal==0.4.0;
SYSTEM_VERSION_COMPAT=0 pip install tensorflow-federated==0.20.0
```

we need to use `SYSTEM_VERSION_COMPAT=0`, otherwise an error will occur.
More info regarding the error can be found:
* https://developer.apple.com/forums/thread/683757
* https://developer.apple.com/metal/tensorflow-plugin/


#### (Deprecagted) install tensorflow macos 2.5.0
Install the tensorflow-macos version 2.5.0 and tensorflow-metal==0.2.0
```
SYSTEM_VERSION_COMPAT=0 pip install tensorflow-macos==2.5.0 tensorflow-federated==0.19.0 tensorflow-metal==0.2.0 
```
notice:\
tensorflow-metal 0.3.0 doesn't work with tensorflow-federated 0.19.0, since tensorflow-federated 0.19.0 depends on tf 2.5

alternatively you can installed the tensorflow-macos==2.7.0 without the tensorflow-federated
```
SYSTEM_VERSION_COMPAT=0 pip install tensorflow-macos==2.7.0 tensorflow-metal==0.3.0 
SYSTEM_VERSION_COMPAT=0 pip install tensorflow-federated==0.19.0
```

```
SYSTEM_VERSION_COMPAT=0 pip install tensorflow-macos==2.8.0 tensorflow-metal==0.4.0;
SYSTEM_VERSION_COMPAT=0 pip install tensorflow-federated==0.20.0
```

this will install tensorflow 2.7.0 version,
You may see an dependency warning:
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
tensorflow-macos 2.7.0 requires tensorflow-estimator<2.8,~=2.7.0rc0, but you have tensorflow-estimator 2.5.0 which is incompatible.
```
but it will work with the tensorflow-federated

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



