# Install the packages
```
source <VENV>/bin/activate
python3 -m pip install -r <path_to_this_module_folder>/requirements.txt
```
# Issue Matplotlib plot not show in vscode
```python
from matplotlib.pyplot as plt

...

plt.show()
```
Only when you added plt.show() in your vscode program, the matplotlib plot will be show.

<!--
Open VSCode Settings (JSON) and add the following line to your user setting.json file:
```json
{
    "python.defaultInterpreterPath": "<path>/<venv_name>/bin/python3",
    "terminal.integrated.inheritEnv": true
}
```
Reference:
* https://stackoverflow.com/questions/61757455/matplotlib-figure-wont-show-when-python-is-run-from-vs-code-integrated-terminal
-->