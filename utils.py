from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score
from tensorflow.keras.utils import plot_model

# Normalización de datos
def normalizar_datos(X, tipo='minmax'):
    if tipo == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    X_normalizado = scaler.fit_transform(X)
    return X_normalizado, scaler

# Evaluación para regresión
def evaluar_regresion(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return {'MSE': mse, 'MAE': mae}

# Evaluación para clasificación
def evaluar_clasificacion(y_true, y_pred):
    y_pred_bin = (y_pred > 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred_bin)
    prec = precision_score(y_true, y_pred_bin)
    recall = recall_score(y_true, y_pred_bin)
    return {'Accuracy': acc, 'Precision': prec, 'Recall': recall}

# Visualización del modelo
def guardar_modelo(modelo, nombre_archivo='modelo.png'):
    plot_model(modelo, to_file=nombre_archivo, show_shapes=True, show_layer_names=True)
