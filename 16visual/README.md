# Introduction into visualization

## Install packages 
```
source ~/VENV/vis3.10/bin/activate
python3 -m pip install -r requirements.txt --no-cache
deactivate
```

## Plots
* Calibration plot for finding the biases through the data slices by comparing the label with model prediction

## Remove all package from venv
```
python3 -m pip freeze | xargs pip uninstall -y
python3 -m pip list
```

Reference:
* Calibration Plot for finding bias https://cran.r-project.org/web/packages/predtools/vignettes/calibPlot.html
* seaborn jointplot https://seaborn.pydata.org/tutorial/regression.html#plotting-a-regression-in-other-contexts
* Signal with noise: https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python/26181710#26181710
