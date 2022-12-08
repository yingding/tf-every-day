# About

## Install packages 
```
python3 -m pip install -r requirements.txt
```
## Remove all package from venv
```
python3 -m pip freeze | xargs pip uninstall -y
python3 -m pip list
```

## Issue
zenml is depending on ml-metadata, and which has no aarch64 package
* https://github.com/google/ml-metadata/issues/143