from collections import Counter
from Otro.Utilidades.Ejercicios.EvaluarEjericios import *
import cv2
import mediapipe as mp

def evaluarEjercicio(nombreEjercicio, keypoints_cuerpo, landmarks, ancho, altura,zonaError, frame_rgb):
    marcas_error_global = []

    def marcar_error(marcas):
        for marca in marcas:
            if marca[0] == "punto":
                _, x, y = marca
                cv2.circle(frame_rgb, (x, y), 8, (0, 0, 255), -1)
            elif marca[0] == "linea":
                _, x1, y1, x2, y2 = marca
                cv2.line(frame_rgb, (x1, y1), (x2, y2), (0, 0, 255), 3)

    match nombreEjercicio:
        case "squat":
            marcas_error_global = veredicto_squat(keypoints_cuerpo, landmarks, ancho, altura,zonaError)
            marcar_error(marcas_error_global)
        case "barbell biceps curl":
            marcas_error_global = veredictoCurl_biceps(keypoints_cuerpo, landmarks, ancho, altura,zonaError)
            marcar_error(marcas_error_global)
        case "pull up":
            marcas_error_global = veredicto_pullup(keypoints_cuerpo, landmarks, ancho, altura,zonaError)
            marcar_error(marcas_error_global)
"""   
def predecir_ejercicio(keypoints,model, le):
    if len(keypoints) == 0:
        print("No se detectaron poses.")
        return "Desconocido"

    print("Extrayendo keypoints...")
    X = np.array(keypoints)

    print("Realizando predicciones...")
    confianza = 0.0
    preds = model.predict(X)
    confianza = max(preds[0])

    print(f"Confianza de la predicción: {confianza:.2f}")

    if confianza < 0.6:
        return "sin_deteccion"

    clases_pred = np.argmax(preds, axis=1)

    clase_mayoritaria = Counter(clases_pred).most_common(1)[0][0]
    ejercicio = le.inverse_transform([clase_mayoritaria])[0]

    return ejercicio
    
"""

# ---------------- ANGULO ----------------
def calcular_angulo(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    den = np.linalg.norm(ba) * np.linalg.norm(bc)
    if den == 0:
        return 0

    cos_angulo = np.dot(ba, bc) / den
    cos_angulo = np.clip(cos_angulo, -1.0, 1.0)

    angulo = np.arccos(cos_angulo)

    return np.degrees(angulo)


# ---------------- ANGULOS IMPORTANTES ----------------
def obtener_angulos(landmarks):

    def p(idx):
        return [landmarks[idx].x, landmarks[idx].y]

    mp_pose = mp.solutions.pose

    angulos = {}

    angulos["left_knee"] = calcular_angulo(
        p(mp_pose.PoseLandmark.LEFT_HIP.value),
        p(mp_pose.PoseLandmark.LEFT_KNEE.value),
        p(mp_pose.PoseLandmark.LEFT_ANKLE.value)
    )

    angulos["right_knee"] = calcular_angulo(
        p(mp_pose.PoseLandmark.RIGHT_HIP.value),
        p(mp_pose.PoseLandmark.RIGHT_KNEE.value),
        p(mp_pose.PoseLandmark.RIGHT_ANKLE.value)
    )

    angulos["left_elbow"] = calcular_angulo(
        p(mp_pose.PoseLandmark.LEFT_SHOULDER.value),
        p(mp_pose.PoseLandmark.LEFT_ELBOW.value),
        p(mp_pose.PoseLandmark.LEFT_WRIST.value)
    )

    angulos["right_elbow"] = calcular_angulo(
        p(mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
        p(mp_pose.PoseLandmark.RIGHT_ELBOW.value),
        p(mp_pose.PoseLandmark.RIGHT_WRIST.value)
    )

    angulos["left_hip"] = calcular_angulo(
        p(mp_pose.PoseLandmark.LEFT_SHOULDER.value),
        p(mp_pose.PoseLandmark.LEFT_HIP.value),
        p(mp_pose.PoseLandmark.LEFT_KNEE.value)
    )

    angulos["right_hip"] = calcular_angulo(
        p(mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
        p(mp_pose.PoseLandmark.RIGHT_HIP.value),
        p(mp_pose.PoseLandmark.RIGHT_KNEE.value)
    )

    return list(angulos.values())

def predecir_ejercicio(keypoints, model, le, scaler):

    if len(keypoints) < 20:
        return "sin_deteccion"

    ventana = np.array(keypoints[-20:], dtype=np.float32)

    kp = ventana  # (frames, 99)

    angulos_frames = []

    for frame in ventana:

        puntos = frame.reshape(33,3)

        class LM:
            def __init__(self,x,y):
                self.x = x
                self.y = y

        landmarks = [LM(p[0], p[1]) for p in puntos]

        ang = obtener_angulos(landmarks)

        angulos_frames.append(ang)

    ang = np.array(angulos_frames)  # (frames, 6)

    features = np.concatenate([

        kp.mean(axis=0),
        kp.std(axis=0),

        ang.mean(axis=0),
        ang.std(axis=0),

        ang.min(axis=0),
        ang.max(axis=0)

    ])

    X = features.reshape(1,-1)

    print("Shape features:", X.shape)  # debe dar (1,222)

    X = scaler.transform(X)

    preds = model.predict(X, verbose=0)[0]

    clase_idx = np.argmax(preds)
    confianza = preds[clase_idx]

    print(f"Confianza: {confianza:.2f}")

    if confianza < 0.6:
        return "sin_deteccion"

    return le.inverse_transform([clase_idx])[0]