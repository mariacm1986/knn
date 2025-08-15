# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 19:17:31 2025

@author: Maria G
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Cargar los datos
dataset = pd.read_csv('datos_knn.csv')

# Separa el dataset en variables independientes y
# variables dependientes
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Dividir en dataset en conjunto de entrenamiento y en 
# conjunto de testing.
# Dividir en entrenamiento y prueba.
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=0)

# Normalizar o escalar los valores
scalarDatos = StandardScaler()
x_train_escalado = scalarDatos.fit_transform(x_train)

# Entrenar el modelo KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train_escalado, y_train)

# Nuevo
nuevo_dato = [[1.70, 120]]
nuevo_dato_scaler = scalarDatos.transform(nuevo_dato)
prediccion = knn.predict(nuevo_dato_scaler)

# Graficar datos originales(no estan escalados) y voy a graficar
# el nuevo punto(figsize=(8,6))

# Colorear puntos por clase (Deportista, y No Deportista)
for clase in y.unique():
    datos_clase = dataset[dataset[dataset.columns[-1]] == clase]
    plt.scatter(
        datos_clase.iloc[:, 0],
        datos_clase.iloc[:, 1],
        label=clase
    
    )

# Agregar  dato predecido  (nuevo_dato = [[1.70, 120]])
plt.scatter(1.70, 120, color="black", marker='X', s=100, label="Nuevo Dato(1.70,120)")

# Personalizar este grafico
plt.xlabel(dataset.columns[0])
plt.ylabel(dataset.columns[1])
plt.title(f"Clasificacion KNN - Prediccion:{prediccion[0]}")
plt.legend()
plt.grid(True)
plt.show()