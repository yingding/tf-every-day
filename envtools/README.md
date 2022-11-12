# About this project

Provides zsh scripts to setup a python venv on your macbook quickly

## conditions
- home brew shall be installed on your macosx system
- the python3 version 3.8, 3.9 or 3.10 is installed on your macosx system with home brew

## envtool
Constains
- call `source shell_info.sh` to determine whether your have a zsh activated on your macosx
- call `create_env.sh -p <path/envname> -v <3.8|3.9|3.10>` to create a python venv using python package from home brew

## example call:
```console
source create_env.sh -p ~/VENV/tfx3.10 -v 3.10
```
creates a python3 venv of `python3.10` at path `~/VENV/tfx3.10` 

## Remove the venv
Use `rm` to remove the python venv at the path created.
```console
rm -r <path/envname>
```

## ZSH Reference
* run zsh script: https://rowannicholls.github.io/bash/intro/passing_arguments.html
* zsh condition: https://zsh.sourceforge.io/Doc/Release/Conditional-Expressions.html
* save user zsh input to a variable $REPLY: https://stackoverflow.com/questions/15174121/how-can-i-prompt-for-yes-no-style-confirmation-in-a-zsh-script/15174634#15174634
* zsh test file exists -f at the path location: https://stackoverflow.com/questions/7522712/how-can-i-check-if-a-command-exists-in-a-shell-script/7522866#7522866
* zsh test variable length zero -z: https://rowannicholls.github.io/bash/intro/passing_arguments.html
