from tkinter import filedialog
import cv2
import mediapipe as mp
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from collections import Counter
from EvaluarEjericios import evaluar_sentadilla
mp_pose = mp.solutions.pose
mediapipe_keypoints = {
    0: "nose",
    1: "left_eye_inner",
    2: "left_eye",
    3: "left_eye_outer",
    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",
    7: "left_ear",
    8: "right_ear",
    9: "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",
    18: "right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle"
}

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



def convertir_landmarks_a_diccionario(results):
    puntos_xy = {}
    if results.pose_landmarks:
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            nombre = mediapipe_keypoints.get(idx, f"punto_{idx}")
            puntos_xy[nombre] = [landmark.x, landmark.y]
    return puntos_xy

def main( model, le):
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])

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
    
    if ejercicio == "squat":
        resultado = evaluar_sentadilla(keypoints_cuerpo)
        print("Resultado de la evaluación de la sentadilla:", resultado)
    elif ejercicio == "curl_biceps":
        from EvaluarEjericios import evaluar_curl_biceps
        resultado = evaluar_curl_biceps(keypoints_cuerpo)
        print("Resultado de la evaluación del curl de bíceps:", resultado)
    else:
        print("Ejercicio no soportado para evaluación automática.")
    
    print(f"Ejercicio detectado en el video: {ejercicio}")
    return False

import numpy as np

def calcular_angulo(a, b, c):
    """Calcula el ángulo en el punto b entre a y c"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return angle

if __name__ == "__main__":
    main()
