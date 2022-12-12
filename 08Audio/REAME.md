# About this repository


## Install packages 
```
python3 -m pip install -r requirements.txt --no-cache
```
## Remove all package from venv
```
python3 -m pip freeze | xargs pip uninstall -y
python3 -m pip list
```

# Reference:
* Audio Data Preparation and Augmentation in TensorFlow: https://www.tensorflow.org/io/tutorials/audio