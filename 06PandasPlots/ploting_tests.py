import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DayLocator, DateFormatter
from pandas import DatetimeIndex

# dtype='datetime64[ns]'
# https://stackoverflow.com/questions/26526230/plotting-datetimeindex-on-x-axis-with-matplotlib-creates-wrong-ticks-in-pandas-0

# date_index: DatetimeIndex = pd.date_range(start="1/1/2000", periods=200)

series = pd.Series(np.random.randn(200),
    index=pd.date_range(start="1/1/2000", periods=200))

fig, ax = plt.subplots() 
# important to get the ax and fig, to allow xticks rotation and formatting
py_datetime_idx = series.index.to_pydatetime() 
# convert pandas datetime64 type of DatetimeIndex object to python datetime object
# since matplotlib doesn't work with datetime64 
ax.bar(py_datetime_idx, series.values) 
# use the subplot bar chart, to allow xaxis settings
ax.set_xlim(py_datetime_idx[0], py_datetime_idx[-1]) 
# set the max and min of xticks to match the series index
ax.set_xticks(series.index.to_pydatetime()) 
# set the xticks to use python datetime instead of dateime64 from pandas
ax.xaxis.set_major_locator(DayLocator(interval=10))
 # set xtick to be show by every 10 day
ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d')) # set the xtick datetime format
plt.xticks(rotation = 90) # rotate the xtick 90 degree
plt.show()

# ax = series.plot(kind="bar")

# ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
# plt.plot([1,2,3], [10,20,30])
# plt.show()

# print(series.head(3))