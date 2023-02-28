# Introduction into SHAP

SHAP trains an additional model based on the observation of custom model behaviour to determine how black box custom model making decision regarding the input features.

## Install packages 
```
python3 -m pip install -r requirements.txt --no-cache
```
## Remove all package from venv
```
python3 -m pip freeze | xargs pip uninstall -y
python3 -m pip list
```

Reference:
* feature importance post-hoc model agnostic explanation with SHAP: https://shap.readthedocs.io/en/latest/
* SHAP examples: https://github.com/slundberg/shap
* Intro into SHapley Additive exPlainations: https://christophm.github.io/interpretable-ml-book/shap.html
