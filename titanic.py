import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('datasets/titanic3.csv', dtype={'age':np.float64})

print(data.head())