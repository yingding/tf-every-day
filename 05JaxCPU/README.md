# Install the jax on cpu

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
/usr/local/opt/python@3.8/bin/python3 -m venv ~/jax3.8
```
The path `~/jax3.8` is where you would like to have your virtual env

## Reference:
* Google/JAX Github: https://github.com/google/jax
* Jax Quick Start Doc: https://jax.readthedocs.io/en/latest/notebooks/quickstart.html


## Install jax
```
source ~/jax3.8/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade "jax[cpu]"
```