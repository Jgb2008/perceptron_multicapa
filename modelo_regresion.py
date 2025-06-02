# modelo_regresion.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def crear_modelo_regresion(input_shape, optimizer='adam'):
    model = Sequential([
        Dense(16, activation='relu', input_shape=(input_shape,)),
        Dense(8, activation='relu'),
        Dense(1, activation='linear')  # regresión → salida continua
    ])
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', 'mse']
    )
    return model
