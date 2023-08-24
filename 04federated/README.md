# TensorFlow Federated

## Creat VENV
use the `create_env.sh` script to create a venv on macosx

```shell
VERSION=3.9;
PREFIX=tff;
ENV_NAME="${PREFIX}${VERSION}";
ENV_ROOT="$HOME/VENV";
source ./envtools/create_env.sh -p ${ENV_ROOT}/${ENV_NAME} -v $VERSION
```

## Install packages 
```shell
VERSION=3.9;
PREFIX=tff;
ENV_NAME="${PREFIX}${VERSION}";
PROJ="./04federated";
ENV_ROOT="$HOME/VENV";
source ${ENV_ROOT}/${ENV_NAME}/bin/activate;
cd $PROJ;
python3 -m pip install -r requirements.txt --no-cache
```

```shell
# python3 -m pip install tensorflow-federated==0.48.0 tensorflow-privacy==0.8.10 --no-deps
python3 -m pip install tensorflow-federated==0.34.0 --no-deps
```
Note: `tensorflow-federated==0.34.0` do not require compression

## Add a jupyter notebook kernel to VENV
```shell
VERSION=3.9;
PREFIX=tff;
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
PREFIX=tff;
ENV_NAME="${PREFIX}${VERSION}";
jupyter kernelspec uninstall -y ${ENV_NAME}
```

## Remove all package from venv
```shell
VERSION=3.9;
PREFIX=tff;
ENV_NAME="${PREFIX}${VERSION}";
ENV_ROOT="$HOME/VENV"
source ${ENV_ROOT}/${ENV_NAME}/bin/activate;
python3 -m pip freeze | xargs pip uninstall -y
python3 -m pip list
```