# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 17:39:05 2025

@author: Maria G
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

# Título en Streamlit
st.title("Clasificación KNN con Matriz de Confusión y Precisión")

# Cargar los datos
dataset = pd.read_csv('datos_knn.csv')
st.write("### Datos cargados")
st.dataframe(dataset)

# Variables independientes y dependientes
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Dividir en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Escalar datos
scalarDatos = StandardScaler()
x_train_escalado = scalarDatos.fit_transform(x_train)
x_test_escalado = scalarDatos.transform(x_test)

# Entrenar modelo
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train_escalado, y_train)

# Predicciones sobre datos de prueba
y_pred = knn.predict(x_test_escalado)

# Calcular precisión
precision = accuracy_score(y_test, y_pred)
st.write(f"### Precisión del modelo: {precision:.2f}")

# Matriz de confusión
matriz_conf = confusion_matrix(y_test, y_pred)
st.write("### Matriz de Confusión")
fig_conf, ax_conf = plt.subplots()
sns.heatmap(matriz_conf, annot=True, fmt="d", cmap="Blues", xticklabels=knn.classes_, yticklabels=knn.classes_)
plt.xlabel("Predicción")
plt.ylabel("Valor real")
st.pyplot(fig_conf)

# Nuevo dato
nuevo_dato = [[1.70, 120]]
nuevo_dato_scaler = scalarDatos.transform(nuevo_dato)
prediccion = knn.predict(nuevo_dato_scaler)

# Gráfico de clasificación
fig, ax = plt.subplots(figsize=(8, 6))
for clase in y.unique():
    datos_clase = dataset[dataset[dataset.columns[-1]] == clase]
    ax.scatter(
        datos_clase.iloc[:, 0],
        datos_clase.iloc[:, 1],
        label=clase
    )

# Punto nuevo
ax.scatter(1.70, 120, color="black", marker='X', s=100, label="Nuevo Dato (1.70,120)")

# Personalizar gráfico
ax.set_xlabel(dataset.columns[0])
ax.set_ylabel(dataset.columns[1])
ax.set_title(f"Clasificación KNN - Predicción: {prediccion[0]}")
ax.legend()
ax.grid(True)
st.pyplot(fig)