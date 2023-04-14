# Introduction into pytorch

## Install packages 
```
source ~/VENV/torch3.10/bin/activate
python3 -m pip install -r requirements.txt --no-cache
deactivate
```

## Remove all package from venv
```
python3 -m pip freeze | xargs pip uninstall -y
python3 -m pip list
```

Reference:
* learning the pytorch basics: https://pytorch.org/tutorials/beginner/basics/intro.html
* https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
* https://pytorch.org/
* https://pytorch.org/docs/master/notes/mps.html
* https://developer.apple.com/metal/pytorch/
