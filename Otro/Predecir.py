import cv2
import mediapipe as mp
import numpy as np
from collections import Counter
from Utilidades import convertir_landmarks_a_diccionario
mp_pose = mp.solutions.pose


def extraer_keypoints(video_path):
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose(static_image_mode=False)
    keypoints = []
    keypoints_cuerpo = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            puntos = []
            for lm in results.pose_landmarks.landmark:
                puntos.extend([lm.x, lm.y, lm.z])

            keyCuerpo = convertir_landmarks_a_diccionario(results)
            keypoints_cuerpo.append(keyCuerpo)

            keypoints.append(puntos)
    cap.release()
    pose.close()
    return keypoints, keypoints_cuerpo

def predecirVideo( model, le,video_path=None):

    print("Extrayendo keypoints...")
    X,keypoints_cuerpo = extraer_keypoints(video_path)
    if len(X) == 0:
        print("No se detectaron poses en el video.")
        return

    X = np.array(X)

    print("Realizando predicciones...")
    preds = model.predict(X)
    clases_pred = np.argmax(preds, axis=1)

    clase_mayoritaria = Counter(clases_pred).most_common(1)[0][0]

    # Obtener la etiqueta original con el label encoder
    ejercicio = le.inverse_transform([clase_mayoritaria])[0]

    print(f"Ejercicio detectado en el video: {ejercicio}")
    return ejercicio

import numpy as np

