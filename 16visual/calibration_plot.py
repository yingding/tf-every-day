"""
Generate noice
https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python/26181710#26181710
"""

import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt 

# simulate labels
# labels = np.linspace(-1, 1, 100) # 100 values between -1 and 1, uniform distribution
# labels = np.random.normal(0, 1, 100) # standard normal
labels = np.random.standard_normal(100)
# generate noise
# noise = np.random.normal(0, 1, 100) # standard normal with 100 points
noise = np.random.normal(0, .2, 100) # normal distribution with mean 0 and standard deviation 0.2 with 100 points
predictions = labels + noise

data = np.transpose(np.array([labels, predictions]))
df_data = pd.DataFrame(data, columns=["labels", "predictions"])

sns.jointplot(data=df_data, x="labels", y="predictions", kind="reg")
plt.show()