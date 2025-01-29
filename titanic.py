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


def createDummies(df1, var_name):
    #Se extraen las variables dummies de la columano 'sex'(0 o 1; True o False)
    dummies = pd.get_dummies(df1[var_name], prefix=1)
    #Se elimina la columna 'sex'
    #Axis = 1 hace referencia a las columnas y Axis= 0, a las filas
    df1 = df1.drop([var_name], axis=1)
    #Se concatena la la nueva columna df_dummies
    df1 = pd.concat([df1, dummies], axis=1)

    return df1

df = createDummies(df, 'sex')

#Regla de Sturges (division proporcional en un histograma): 1+log2(n) -> n = muestra
k = int(np.ceil(1 + np.log2(len(df['age']))))
#Histograma de las edades y sex correspondientes
plt.hist(df['age'], bins=k)
plt.title('Histograma de las edades')
plt.show()