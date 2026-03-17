import threading
import cv2
import mediapipe as mp
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
from Otro.Modelo.EntrenarModeloNew import main as entrenar_modelo

mp_pose = mp.solutions.pose


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


# ---------------- EXTRACCION ----------------
def extraer_keypoints(video_path):

    cap = cv2.VideoCapture(video_path)

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    keypoints = []
    angulos = []

    paused = False

    while cap.isOpened():

        if not paused:

            ret, frame = cap.read()

            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = pose.process(image_rgb)

            if results.pose_landmarks:

                h, w, _ = frame.shape
                landmarks = results.pose_landmarks.landmark

                puntos = []

                for lm in landmarks:

                    puntos.extend([lm.x, lm.y, lm.z])

                    x = int(lm.x * w)
                    y = int(lm.y * h)

                    cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)

                keypoints.append(puntos)

                angulos_frame = obtener_angulos(landmarks)

                angulos.append(angulos_frame)

        cv2.imshow("Keypoints", frame)

        key = cv2.waitKey(10) & 0xFF

        if key == 27:
            break

        elif key == 32:
            paused = not paused

    cap.release()
    pose.close()
    cv2.destroyAllWindows()

    return keypoints, angulos


# ---------------- DATASET ----------------
def subir_video():

    X = []
    y = []

    extensiones_video = [".mp4", ".avi", ".mov", ".mkv"]

    carpeta = filedialog.askdirectory(title="Selecciona una carpeta con videos")

    if not carpeta:
        return

    videos = [
        os.path.join(carpeta, f)
        for f in os.listdir(carpeta)
        if os.path.splitext(f)[1].lower() in extensiones_video
    ]

    print("Videos encontrados:", videos)

    for ruta_video in videos:

        nombre = os.path.splitext(os.path.basename(ruta_video))[0]

        partes = nombre.split("_")

        if len(partes) < 3:

            print(f"⚠️ Nombre inválido: {ruta_video}")
            continue

        ejercicio = partes[0]
        etiqueta = partes[1]

        clase = f"{ejercicio}_{etiqueta}"

        print("Procesando:", ruta_video, "|", clase)

        kp, ang = extraer_keypoints(ruta_video)

        if len(kp) == 0:
            print("⚠️ No se detectaron keypoints")
            continue

        kp = np.array(kp)
        ang = np.array(ang)

        features = np.concatenate([

            kp.mean(axis=0),
            kp.std(axis=0),

            ang.mean(axis=0),
            ang.std(axis=0),

            ang.min(axis=0),
            ang.max(axis=0)

        ])

        X.append(features)
        y.append(clase)

    datos_dir = "datos"
    os.makedirs(datos_dir, exist_ok=True)

    x_path = os.path.join(datos_dir, "X.npy")
    y_path = os.path.join(datos_dir, "y.npy")

    X = np.array(X)
    y = np.array(y)
    if os.path.exists(x_path) and os.path.exists(y_path):

        X_existente = np.load(x_path)
        y_existente = np.load(y_path)

        if X_existente.ndim == 1:
            X_existente = X_existente.reshape(1, -1)

        X = np.concatenate([X_existente, X], axis=0)
        y = np.concatenate([y_existente, y], axis=0)

    np.save(x_path, X)
    np.save(y_path, y)

    print("Dataset final:")
    print("X:", X.shape)
    print("y:", y.shape)

    entrenar_modelo()


# ---------------- THREAD ----------------
def iniciar_busqueda_en_hilo():

    hilo = threading.Thread(target=subir_video)
    hilo.start()


# ---------------- GUI ----------------
def main():

    ventana = tk.Tk()
    ventana.title("Entrenamiento IA - WorkoutEye")
    ventana.geometry("400x200")

    btn_video = tk.Button(
        ventana,
        text="Cargar videos y entrenar IA",
        command=iniciar_busqueda_en_hilo
    )

    btn_video.pack(pady=40)

    ventana.mainloop()


if __name__ == "__main__":
    main()