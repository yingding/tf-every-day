"""
based on example
https://nixtla.github.io/statsforecast/examples/getting_started_short.html
"""
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
import pandas as pd

df = pd.read_csv('https://datasets-nixtla.s3.amazonaws.com/air-passengers.csv')
print(df.head())
level = [95] # [90]

sf = StatsForecast(
    models = [AutoARIMA(season_length= 12)],
    freq = 'M'
)

sf.fit(df)
forecast_df = sf.predict(h=12, level=level)

forecast_df.tail()

df["ds"]=pd.to_datetime(df["ds"])
sf.plot(df, forecast_df, level=level)