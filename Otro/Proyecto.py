import threading
import cv2
import mediapipe as mp
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog  
from collections import defaultdict

mp_pose = mp.solutions.pose

def extraer_keypoints(video_path):
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose(static_image_mode=False)
    keypoints = []
    angulos_rodilla = []
    angulos_desicion = []
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
                for lm in results.pose_landmarks.landmark:
                    puntos.extend([lm.x, lm.y, lm.z])
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(frame, (x, y), 6, (255, 0, 0), -1)

                keypoints.append(puntos)

                cadera = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                rodilla = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                tobillo = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                
                angulo = calcular_angulo(cadera, rodilla, tobillo)
                if 70 <= angulo:
                    angulos_desicion.append("buena")
                    #print("Buena")
                else:
                    #print("Mala")
                    #paused = not paused
                    angulos_desicion.append("mala")
                
                angulos_rodilla.append(angulo)

        cv2.imshow('Keypoints', frame)

        key = cv2.waitKey(10) & 0xFF
        if key == 27:  # ESC para salir
            break
        elif key == 32:  # Barra espaciadora para pausar/reanudar
            paused = not paused

    cap.release()
    pose.close()
    cv2.destroyAllWindows()
    return keypoints, angulos_rodilla, angulos_desicion
    
def calcular_angulo(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cos_angulo = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angulo = np.arccos(np.clip(cos_angulo, -1.0, 1.0))
    return np.degrees(angulo) #Numero decimal que es el angulo

def subir_video():
    global ruta_video
    #ruta_video = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
    #if not ruta_video:
    #   return
    
    X = []
    y = []
    Angulos_rodilla = []
    Angulos_desicion = []
    
    extensiones_video = ['.mp4', '.avi', '.mov', '.mkv']
    carpeta = filedialog.askdirectory(title="Selecciona una carpeta con videos")
    if carpeta:
        print("Carpeta seleccionada:", carpeta)
        videos = [os.path.join(carpeta, f) for f in os.listdir(carpeta)
                if os.path.splitext(f)[1].lower() in extensiones_video]
        #videos_dict = {
        #   os.path.splitext(os.path.basename(ruta))[0].split('_')[0]: ruta
        #  for ruta in videos
        #}
        videos_dict = defaultdict(list)
        for ruta in videos:
            clave = os.path.splitext(os.path.basename(ruta))[0].split('_')[0]
            videos_dict[clave].append(ruta)
        print("Videos encontrados:", videos_dict)
        if not videos_dict:
            print("No se encontraron videos en la carpeta seleccionada.")
            return
        print("Procesando videos...")
        for etiqueta, rutas_videos in videos_dict.items():
            for ruta_video in rutas_videos:
                print(f"Procesando video: {ruta_video}")
                kp, angulos, desiciones = extraer_keypoints(ruta_video)

                Angulos_rodilla.extend(angulos)
                Angulos_desicion.extend(desiciones) 
                print(etiqueta)
                X.extend(kp)
                y.extend([etiqueta] * len(kp))

        '''for etiqueta,video in videos_dict.items():
            #x tiene que ser una lista de listas, tiene listas de keypoints cada frame es una lista, entonces cada frame es una lista de keypoints
            #y tiene que ser una lista de etiquetas
            kp,angulos,desiciones = extraer_keypoints(video)

            Angulos_rodilla.extend(angulos)
            Angulos_desicion.extend(desiciones) 

            X.extend(kp)
            y.extend([etiqueta] * len(kp))  # Cambia "etiqueta" por la etiqueta correspondiente al video
    '''
        X = np.array(X)
        y = np.array(y)
        Angulos_desicion = np.array(Angulos_desicion)
        Angulos_rodilla = np.array(Angulos_rodilla)

        os.makedirs("datos", exist_ok=True)
        np.save("datos/X.npy", X)
        np.save("datos/y.npy", y)

        np.save("datos/angulos_rodilla.npy", Angulos_rodilla)
        np.save("datos/angulos_desicion.npy", Angulos_desicion)
        
        print(f"Valores extraídos: {X.shape}, {y.shape}")

        print(f"Datos guardados en la carpeta 'datos'. Tamaño X: {X.shape}, y: {y.shape}")

def iniciar_busqueda_en_hilo(): 
    hilo = threading.Thread(target=subir_video)
    hilo.start()

def main():

    ventana = tk.Tk()
    ventana.title("Ventana Principal")
    ventana.geometry("500x500")

    btn_video = tk.Button(ventana, text="Video", command=iniciar_busqueda_en_hilo)
    btn_video.pack(pady=10)

    ventana.mainloop()

if __name__ == "__main__":
    main()
