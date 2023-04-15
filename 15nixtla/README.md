# Introduction into Nixtla statsforecast

Time Series forecast python lib

the alternative is Darts
* https://unit8co.github.io/darts/

online book of time series forecast in R by Prof. Hyndman:
* https://otexts.com/fpp3/


## Install packages 
```
source ~/VENV/series3.10/bin/activate
python3 -m pip install -r requirements.txt --no-cache
deactivate
```

## Remove all package from venv
```
python3 -m pip freeze | xargs pip uninstall -y
python3 -m pip list
```

## Referece:
* https://github.com/Nixtla/statsforecast
* https://unit8co.github.io/darts/
* https://www.sktime.net/en/latest/
