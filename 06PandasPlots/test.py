import pandas as pd
import os

print(os.getcwd())
df = pd.read_csv('06PandasPlots/population_growth_tiny.csv', index_col=0)
print(df)

df.to_csv('06PandasPlots/test.csv')