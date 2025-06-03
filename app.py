import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD

# Cargar el dataset
@st.cache_data
def load_data():
    df = pd.read_csv("dataset/diamonds.csv")
    return df

df = load_data()

# Mostrar los primeros datos
st.title("MLP - Predicción y Clasificación de Precios de Diamantes")
st.write("Vista previa del dataset:")
st.dataframe(df.head())

# Definir X y y para regresión y clasificación
X = df[['carat', 'depth', 'table']].values

# Salida para regresión
y_reg = df['price'].values

# Salida para clasificación (precio > 350)
y_clf = (df['price'] > 350).astype(int).values

# Normalizar X
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Separar datos
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_scaled, y_reg, test_size=0.2, random_state=42)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_scaled, y_clf, test_size=0.2, random_state=42)

# Sidebar - opciones del modelo
st.sidebar.title("Configuración del modelo")
opt = st.sidebar.selectbox("Optimizador", ["adam", "sgd"])
loss_reg = st.sidebar.selectbox("Función de pérdida (regresión)", ["mse", "mae"])
loss_clf = st.sidebar.selectbox("Función de pérdida (clasificación)", ["binary_crossentropy"])
epochs = st.sidebar.slider("Épocas", 10, 200, 50)

# Crear modelo
def crear_modelo(tipo='regresion', opt='adam', loss='mse'):
    model = Sequential()
    model.add(Dense(16, input_shape=(X.shape[1],), activation='relu'))
    model.add(Dense(8, activation='relu'))

    if tipo == 'regresion':
        model.add(Dense(1, activation='linear'))
        metrics = ['mae', 'mse']
    else:
        model.add(Dense(1, activation='sigmoid'))
        metrics = ['accuracy', 'Precision', 'Recall']

    optimizer = Adam() if opt == 'adam' else SGD()
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

# Entrenar regresión
if st.button("Entrenar modelo de regresión"):
    model_reg = crear_modelo('regresion', opt, loss_reg)
    model_reg.fit(X_train_r, y_train_r, epochs=epochs, verbose=0)
    y_pred_r = model_reg.predict(X_test_r).flatten()
    mse = mean_squared_error(y_test_r, y_pred_r)
    mae = mean_absolute_error(y_test_r, y_pred_r)
    st.subheader("Resultados de Regresión")
    st.write(f"MAE: {mae:.2f}")
    st.write(f"MSE: {mse:.2f}")

# Entrenar clasificación
if st.button("Entrenar modelo de clasificación"):
    model_clf = crear_modelo('clasificacion', opt, loss_clf)
    model_clf.fit(X_train_c, y_train_c, epochs=epochs, verbose=0)
    y_pred_c = model_clf.predict(X_test_c).flatten()
    y_pred_labels = (y_pred_c > 0.5).astype(int)
    acc = accuracy_score(y_test_c, y_pred_labels)
    prec = precision_score(y_test_c, y_pred_labels)
    rec = recall_score(y_test_c, y_pred_labels)
    st.subheader("Resultados de Clasificación")
    st.write(f"Accuracy: {acc:.2f}")
    st.write(f"Precision: {prec:.2f}")
    st.write(f"Recall: {rec:.2f}")
