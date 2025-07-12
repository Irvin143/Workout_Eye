from fastapi import FastAPI, UploadFile, File, Form
from typing import List
import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
import pickle
from Otro.Utilidades.Ejercicios.UtilEjercicio import predecir_ejercicio
from Otro.Utilidades.Ejercicios.EvaluarEjericios import evaluar_sentadilla, evaluar_curl_biceps, evaluar_pullup
from Otro.Utilidades.Utilidades import convertir_landmarks_a_diccionario

app = FastAPI()

mp_pose = mp.solutions.pose

model = None
le = None

@app.on_event("startup")
def cargar_modelo():
    global model, le
    print("Cargando modelo y etiquetas...")
    model = load_model("datos/modelo_ejercicios.h5")
    with open("datos/labels.pkl", "rb") as f:
        le = pickle.load(f)
    print("Modelo y etiquetas cargados.")

@app.post("/keypoints")
async def detectar_keypoints(
    frames: List[UploadFile] = File(...)
):
    keypoints = []
    keypoints_cuerpo = []
    repeticiones = 0
    zonaError = []
    nombre_ejercicio = "Desconocido"

    with mp_pose.Pose(static_image_mode=True) as pose:
        for file in frames:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                continue

            results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not results.pose_landmarks:
                continue

            puntos = []
            for lm in results.pose_landmarks.landmark:
                puntos.extend([lm.x, lm.y, lm.z])
            keypoints.append(puntos)

            keyCuerpo = convertir_landmarks_a_diccionario(results)
            keypoints_cuerpo.append(keyCuerpo)

    if keypoints:
        nombre_ejercicio = predecir_ejercicio(keypoints, model, le)
        match nombre_ejercicio:
            case "squat":
                zonaError, repeticiones = evaluar_sentadilla(keypoints_cuerpo)
            case "barbell biceps curl":
                zonaError, repeticiones = evaluar_curl_biceps(keypoints_cuerpo)
            case "pull up":
                zonaError, repeticiones = evaluar_pullup(keypoints_cuerpo)
            case _:
                zonaError = []

    return {
        "NombreEjercicio": nombre_ejercicio,
        "Repeticiones": repeticiones,
        "ZonaError": zonaError,
        "TotalFramesProcesados": len(keypoints)
    }
