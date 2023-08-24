# About this repo

this repo is the advanced project of 07tflite, the steps below from the previous project is wrapped in a tfx pipeline.

Following the instruction of On-device Training to 
1. train a mnist model (tf_train.py)
2. convert the trained tf model to tflite using compression (tflite_convert.py)
3. inference the compressed tflite model (tflite_inference.py)
4. retrain tflite model on python, or android device (tflite_retrain.py)
5. inference the retained tflite model from checkpoint (tflite_retrain_inference.py)


## Create a python venv with zsh automation script
<!-- 
```console
pushd envtools
source create_env.sh -p ~/VENV/tfx3.9 -v 3.9
popd
```
-->

```shell
VERSION=3.9;
PREFIX=tfx;
ENV_NAME="${PREFIX}${VERSION}";
ENV_ROOT="$HOME/VENV";
source ./envtools/create_env.sh -p ${ENV_ROOT}/${ENV_NAME} -v $VERSION
```
Note: 
* at the time of writing, the `tfx==1.13.0` only support `python3.9`, NOT `python3.10`

## Install packages
```shell
VERSION=3.9;
PREFIX=tfx;
ENV_NAME="${PREFIX}${VERSION}";
PROJ="./09tfx";
ENV_ROOT="$HOME/VENV";
source ${ENV_ROOT}/${ENV_NAME}/bin/activate;
cd $PROJ;
python3 -m pip install -r requirements.txt --no-cache
```

## Add a jupyter notebook kernel to VENV
```shell
VERSION=3.9;
PREFIX=tfx;
ENV_NAME="${PREFIX}${VERSION}";
ENV_ROOT="$HOME/VENV"
source ${ENV_ROOT}/${ENV_NAME}/bin/activate;
python3 -m pip install --upgrade pip
python3 -m pip install ipykernel
deactivate
```

We need to reactivate the venv so that the ipython kernel is available after installation.
```shell
VERSION=3.9;
PREFIX=tfx;
ENV_NAME="${PREFIX}${VERSION}";
ENV_ROOT="$HOME/VENV"
source ${ENV_ROOT}/${ENV_NAME}/bin/activate;
# ipython kernel install --user --name=${ENV_NAME}
python3 -m ipykernel install --user --name=${ENV_NAME} --display-name ${ENV_NAME}
```
Note: 
* restart the vs code, to select the venv as jupyter notebook kernel 
* name is `${ENV_NAME}`, which is the venv name.

Reference:
* https://ipython.readthedocs.io/en/stable/install/kernel_install.html
* https://anbasile.github.io/posts/2017-06-25-jupyter-venv/

## Remove ipykernel
```shell
# jupyter kernelspec uninstall -y <VENV_NAME>
VERSION=3.9;
PREFIX=tfx;
ENV_NAME="${PREFIX}${VERSION}";
jupyter kernelspec uninstall -y ${ENV_NAME}
```

## Remove all package from venv
```shell
VERSION=3.9;
PREFIX=tfx;
ENV_NAME="${PREFIX}${VERSION}";
ENV_ROOT="$HOME/VENV"
source ${ENV_ROOT}/${ENV_NAME}/bin/activate;
python3 -m pip freeze | xargs pip uninstall -y
python3 -m pip list
```

## Creating TFX pipeline




## Reference:
* Tuturial On-device training: https://www.tensorflow.org/lite/examples/on_device_training/overview#build_a_model_for_on-device_training
* Blog post of On-device training with TensorFlow Lite: https://blog.tensorflow.org/2021/11/on-device-training-in-tensorflow-lite.html
* Another MobileNetV2 example of On-device training on Android: https://github.com/tensorflow/examples/tree/master/lite/examples/model_personalization


# Future Works

* call bash operator, to execute the python scripts in airflow: https://stackoverflow.com/questions/41730297/python-script-scheduling-in-airflow/41731397#41731397
* Run multiple script as subprocesses: https://stackoverflow.com/questions/68980171/running-multiple-python-scripts-with-subprocess-python/68980201#68980201 


