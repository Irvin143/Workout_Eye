from collections import Counter
from EvaluarEjericios import *
import cv2

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
