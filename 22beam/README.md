# Reinforcement Learning

## Creat VENV
use the `create_env.sh` script to create a venv on macosx

```shell
VERSION=3.12;
ENV_NAME="beam${VERSION}";
source ./envtools/create_env.sh -p ~/VENV/${ENV_NAME} -v $VERSION
```

## Install packages 
```shell
VERSION=3.12;
ENV_NAME="beam${VERSION}";
source ~/VENV/${ENV_NAME}/bin/activate;
PROJ="./22beam";
cd $PROJ;
python3 -m pip install -r requirements.txt --no-cache
```

## Add a jupyter notebook kernel to VENV
```shell
VERSION=3.12;
ENV_NAME="beam${VERSION}";
source ~/VENV/${ENV_NAME}/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install ipykernel
deactivate
```

We need to reactivate the venv so that the ipython kernel is available after installation.
```shell
VERSION=3.12;
ENV_NAME="beam${VERSION}";
source ~/VENV/${ENV_NAME}/bin/activate
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
VERSION=3.12;
ENV_NAME="beam${VERSION}";
jupyter kernelspec uninstall -y ${ENV_NAME}
```

## Remove all package from venv
```
python3 -m pip freeze | xargs pip uninstall -y
python3 -m pip list
```

## Apache beam

```shell
python3 -m wordcount --input ./data/input.txt --output ./data/out.txt --runner DirectRunner
```

* Github beam examples https://github.com/apache/beam/tree/master/sdks/python/apache_beam/examples
* https://medium.com/google-cloud/launch-apache-beam-pipeline-locally-using-directrunner-323565b546b2


## Reference:
* Beam WordCount Python quickstart https://beam.apache.org/get-started/quickstart-py/