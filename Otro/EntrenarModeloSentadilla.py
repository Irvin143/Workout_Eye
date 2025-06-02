import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

def main():
    # Cargar datos
    X = np.load("datos/X.npy")
    y = np.load("datos/angulos_desicion.npy")  # Cargamos "buena" o "mala"

    print(f"Datos cargados: X={X.shape}, y={y.shape}")
    
    # Codificar etiquetas "buena"/"mala" a números
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)  # "buena" -> 1, "mala" -> 0 (o viceversa)
    y_categorical = to_categorical(y_encoded)  # Necesario para softmax

    # División entrenamiento/prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

    # Crear modelo
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(y_categorical.shape[1], activation='softmax')  # 2 clases: buena, mala
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Entrenar
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

    # Guardar modelo y codificador
    model.save("modelo_sentadillas_buenas_malas.h5")
    with open("labels_sentadillas.pkl", "wb") as f:
        pickle.dump(le, f)

    print("✅ Modelo entrenado y guardado como modelo_sentadillas_buenas_malas.h5")
    print("✅ Codificador guardado como labels_sentadillas.pkl")

if __name__ == "__main__":
    main()
