import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


class ExerciseModelTrainer:

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.model = RandomForestClassifier(n_estimators=200)
        self.side_encoder = LabelEncoder()
        self.label_encoder = LabelEncoder()

    def load_data(self):
        data = pd.read_csv(self.csv_path)

        # convertir Side a número
        data["Side"] = self.side_encoder.fit_transform(data["Side"])

        # features
        X = data.drop("Label", axis=1)

        # etiqueta
        y = self.label_encoder.fit_transform(data["Label"])

        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self):
        X_train, X_test, y_train, y_test = self.load_data()

        self.model.fit(X_train, y_train)

        predictions = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        print("Accuracy del modelo:", accuracy)

    def save_model(self, model_path="datos/exercise_model.pkl",
                   label_encoder_path="datos/label_encoder.pkl",
                   side_encoder_path="datos/side_encoder.pkl"):

        joblib.dump(self.model, model_path)
        joblib.dump(self.label_encoder, label_encoder_path)
        joblib.dump(self.side_encoder, side_encoder_path)

        print("Modelo guardado correctamente")

trainer = ExerciseModelTrainer("datos/exercise_angles.csv")
trainer.train()
trainer.save_model()
