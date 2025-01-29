import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('datasets/titanic3.csv', dtype={'age':np.float64})

df = data
#Reemplazo de la columana vacias
var = pd.isnull(df['age']).values.ravel().sum() #263 datos nulosñ en la columna 'age'
df['age'] = df['age'].fillna(df['age'].mean())
df['body'] = df['body'].fillna(0)
df['cabin'] = df['cabin'].fillna('Desconocido')
df['boat'] = df['boat'].fillna('Desconocido')
df['home.dest'] = df['home.dest'].fillna('Desconocido')

print(df)