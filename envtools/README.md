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