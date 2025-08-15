# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 20:06:26 2025

@author: Maria G
"""

#✅ Entrenamiento y evaluación del modelo KNN.

#✅ Precisión del modelo en porcentaje.

#✅ Matriz de confusión con seaborn.heatmap.

#✅ Predicción personalizada ingresando Altura y Peso.

#✅ Gráfico visual con el nuevo punto en negro.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import streamlit as st

# --------------------------
# Configurar página
# --------------------------
st.set_page_config(page_title="Clasificación KNN", layout="centered")
st.title("Clasificador KNN: ¿Es deportista o no?")

# --------------------------
# Cargar datos
# --------------------------
datasets = pd.read_csv("datos_knn.csv")

# Variables independientes y dependiente
X = datasets[["Peso", "Altura"]]
y = datasets["Clase"]

# --------------------------
# Dividir en entrenamiento y prueba
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# --------------------------
# Escalar datos
# --------------------------
sc_x = StandardScaler()
X_train_escalado = sc_x.fit_transform(X_train)
X_test_escalado = sc_x.transform(X_test)

# --------------------------
# Entrenar modelo KNN
# --------------------------
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_escalado, y_train)

# --------------------------
# Evaluación del modelo
# --------------------------
y_pred_test = knn.predict(X_test_escalado)
acc = accuracy_score(y_test, y_pred_test)
cm = confusion_matrix(y_test, y_pred_test)

st.subheader("📊 Evaluación del Modelo")
st.write(f"✅ Precisión del modelo: **{acc*100:.2f}%**")

fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=knn.classes_, yticklabels=knn.classes_, ax=ax_cm)
ax_cm.set_xlabel("Predicción")
ax_cm.set_ylabel("Valor Real")
ax_cm.set_title("Matriz de Confusión")
st.pyplot(fig_cm)

# --------------------------
# Formulario de entrada de datos
# --------------------------
st.header("Ingrese los datos para predecir")

col1, col2 = st.columns(2)
with col1:
    Altura = st.number_input("Altura (m)", min_value=1.0, max_value=2.5, step=0.01, format="%.2f")
with col2:
    Peso = st.number_input("Peso (Kg)", min_value=30, max_value=200, step=1)

# --------------------------
# Predicción personalizada
# --------------------------
if st.button("Predecir"):
    nuevo_dato = [[Peso, Altura]]
    nuevo_dato_escalado = sc_x.transform(nuevo_dato)
    prediccion = knn.predict(nuevo_dato_escalado)[0]
    st.success(f"Resultado de la predicción: {prediccion}")

    # --------------------------
    # Graficar datos y nuevo punto
    # --------------------------
    colores = {'deportista': 'blue', 'no deportista': 'red'}

    fig, ax = plt.subplots(figsize=(8, 6))
    for clase in y.unique():
        datos_clase = datasets[datasets["Clase"] == clase]
        ax.scatter(
            datos_clase["Peso"],
            datos_clase["Altura"],
            label=clase,
            color=colores[clase]
        )

    ax.scatter(Peso, Altura, color="black", marker="X", s=100, label="Nuevo Dato")
    ax.set_xlabel("Peso (Kg)")
    ax.set_ylabel("Altura (m)")
    ax.set_title(f"Clasificación KNN - Predicción: {prediccion}")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)