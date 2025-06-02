# app.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from modelo_regresion import crear_modelo_regresion
from modelo_clasificacion import crear_modelo_clasificacion

st.title("🧠 Predicción de Precios de Diamantes con MLP")

modelo_tipo = st.selectbox("Selecciona el tipo de modelo:", ["Regresión", "Clasificación"])
optimizador = st.selectbox("Selecciona el optimizador:", ["adam", "sgd"])
epocas = st.slider("Número de épocas", 10, 500, step=10)
normalizador_tipo = st.radio("Normalización", ["MinMaxScaler", "StandardScaler"])

# Carga y preprocesado
df = pd.read_csv("dataset/diamonds.csv")
X = df[['carat', 'depth', 'table']].values
y = df['price'].values if modelo_tipo == "Regresión" else (df['price'] > 5000).astype(int).values

# Normalización
scaler = MinMaxScaler() if normalizador_tipo == "MinMaxScaler" else StandardScaler()
X_norm = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42)

# Modelo
if modelo_tipo == "Regresión":
    model = crear_modelo_regresion(X.shape[1], optimizer=optimizador)
else:
    act = st.selectbox("Activación de salida", ["sigmoid", "softmax"])
    loss = st.selectbox("Función de pérdida", ["binary_crossentropy", "hinge"])
    model = crear_modelo_clasificacion(X.shape[1], optimizer=optimizador, output_activation=act, loss=loss)

# Entrenamiento
if st.button("Entrenar modelo"):
    with st.spinner("Entrenando..."):
        hist = model.fit(X_train, y_train, epochs=epocas, validation_split=0.2, verbose=0)
        st.success("Entrenamiento completado")

        st.line_chart(hist.history)

        # Evaluación
        loss, *metrics = model.evaluate(X_test, y_test)
        if modelo_tipo == "Regresión":
            st.write(f"**MAE:** {metrics[0]:.2f}")
            st.write(f"**MSE:** {metrics[1]:.2f}")
        else:
            st.write(f"**Accuracy:** {metrics[0]:.2f}")
            st.write(f"**Precision:** {metrics[1]:.2f}")
            st.write(f"**Recall:** {metrics[2]:.2f}")
