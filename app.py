import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam

# Cargar dataset
@st.cache_data
def cargar_datos():
    df = pd.read_csv("dataset/diamonds.csv")
    return df

df = cargar_datos()

st.title("Predicción y Clasificación de Diamantes con Perceptrón Multicapa")
st.write("Dataset cargado:")
st.dataframe(df.head())

# Entrada y salida
X = df[['carat', 'depth', 'table']].values
y_reg = df['price'].values

# Clasificación binaria: si el precio > 350
y_clf = (df['price'] > 350).astype(int).values

# Normalización
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# División de datos
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_scaled, y_reg, test_size=0.2, random_state=42)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_scaled, y_clf, test_size=0.2, random_state=42)

# Sidebar: configuraciones
st.sidebar.header("Configuración del modelo")
opt = st.sidebar.selectbox("Optimizador", ["adam", "sgd"])
loss_reg = st.sidebar.selectbox("Función de pérdida (regresión)", ["mse", "mae"])
loss_clf = st.sidebar.selectbox("Función de pérdida (clasificación)", ["binary_crossentropy"])
epochs = st.sidebar.slider("Épocas de entrenamiento", min_value=10, max_value=200, value=50)

# Función para crear modelo MLP
def crear_modelo(tipo='regresion', optimizador='adam', funcion_loss='mse'):
    model = Sequential()
    model.add(Dense(16, input_shape=(X.shape[1],), activation='relu'))
    model.add(Dense(8, activation='relu'))

    if tipo == 'regresion':
        model.add(Dense(1, activation='linear'))
        metricas = ['mae', 'mse']
    else:
        model.add(Dense(1, activation='sigmoid'))
        metricas = ['accuracy', 'Precision', 'Recall']

    if optimizador == 'adam':
        opt = Adam()
    else:
        opt = SGD()

    model.compile(optimizer=opt, loss=funcion_loss, metrics=metricas)
    return model

# Entrenar modelo de regresión
if st.button("Entrenar modelo de Regresión"):
    model_reg = crear_modelo('regresion', opt, loss_reg)
    history = model_reg.fit(X_train_r, y_train_r, epochs=epochs, verbose=0)
    y_pred_r = model_reg.predict(X_test_r).flatten()
    mse = mean_squared_error(y_test_r, y_pred_r)
    mae = mean_absolute_error(y_test_r, y_pred_r)
    st.subheader("Resultados de regresión")
    st.write(f"MAE: {mae:.2f}")
    st.write(f"MSE: {mse:.2f}")

# Entrenar modelo de clasificación
if st.button("Entrenar modelo de Clasificación"):
    model_clf = crear_modelo('clasificacion', opt, loss_clf)
    history = model_clf.fit(X_train_c, y_train_c, epochs=epochs, verbose=0)
    y_pred_c = model_clf.predict(X_test_c).flatten()
    y_pred_c_bin = (y_pred_c > 0.5).astype(int)
    acc = accuracy_score(y_test_c, y_pred_c_bin)
    prec = precision_score(y_test_c, y_pred_c_bin)
    rec = recall_score(y_test_c, y_pred_c_bin)
    st.subheader("Resultados de clasificación")
    st.write(f"Accuracy: {acc:.2f}")
    st.write(f"Precision: {prec:.2f}")
    st.write(f"Recall: {rec:.2f}")

