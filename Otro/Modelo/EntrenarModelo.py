import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def main():
    # Cargar datos
    X = np.load("datos/X.npy")
    y = np.load("datos/y.npy")

    # Codificar etiquetas
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

    # Crear modelo sencillosa
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(y_categorical.shape[1], activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Entrenar
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

    # Guardar modelo y codificador
    model.save("modelo_ejercicios.h5")

    # Guardar etiquetas para decodificar en predicción
    import pickle
    with open("labels.pkl", "wb") as f:
        pickle.dump(le, f)

    print("Modelo entrenado y guardado como modelo_ejercicios.h5")
    print("Codificador de etiquetas guardado como labels.pkl")

if __name__ == "__main__":
    main()
