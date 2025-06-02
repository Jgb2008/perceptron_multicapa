# modelo_clasificacion.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score

def crear_modelo_clasificacion(input_shape, optimizer='adam', output_activation='sigmoid', loss='binary_crossentropy'):
    model = Sequential([
        Dense(16, activation='relu', input_shape=(input_shape,)),
        Dense(8, activation='relu'),
        Dense(1, activation=output_activation)  # clasificación binaria
    ])
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy', 'Precision', 'Recall']
    )
    return model
