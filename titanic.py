import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('datasets/titanic3.csv', dtype={'age':np.int32})

print(data.head())