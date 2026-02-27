import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle

def main():
    # Cargar datos
    X = np.load("datos/X.npy")
    y = np.load("datos/y.npy")

    print("Dataset cargado:", X.shape, y.shape)

    # Normalizar features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Codificar etiquetas
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    # División entrenamiento / prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Modelo
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X.shape[1],)),
        BatchNormalization(),
        Dropout(0.4),

        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(y_categorical.shape[1], activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Entrenar
    history = model.fit(
        X_train, y_train,
        epochs=40,
        batch_size=32,
        validation_data=(X_test, y_test)
    )

    # Guardar modelo, labels y scaler
    model.save("datos/modelo_ejercicios.h5")
    with open("datos/labels.pkl", "wb") as f:
        pickle.dump(le, f)
    with open("datos/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("✅ Modelo guardado: datos/modelo_ejercicios.h5")
    print("✅ Labels guardados: datos/labels.pkl")
    print("✅ Scaler guardado: datos/scaler.pkl")

if __name__ == "__main__":
    main()