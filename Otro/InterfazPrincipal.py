"""
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import mediapipe as mp
import numpy as np
from collections import Counter

def ventanaPierna():
    nueva_ventana = tk.Toplevel()
    nueva_ventana.title("Pierna")
    nueva_ventana.geometry("300x200")

def ventanaPecho():
    nueva_ventana = tk.Toplevel()
    nueva_ventana.title("Pecho")
    nueva_ventana.geometry("300x200")
    tk.Label(nueva_ventana, text="Esta es la ventana de pecho").pack(pady=20)

def ventanaEspalda():
    nueva_ventana = tk.Toplevel()
    nueva_ventana.title("Espalda")
    nueva_ventana.geometry("300x200")
    tk.Label(nueva_ventana, text="Esta es la ventana de espalda").pack(pady=20)

def detectar_camaras():
    indices = []
    for i in range(5):  # Escanear las primeras 5 cámaras
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            indices.append(str(i))
        cap.release()
    return indices

def mostrar_camara():
    global combo_camaras
    cam_index = int(combo_camaras.get())
    cap = cv2.VideoCapture(cam_index)
    pose = mp.solutions.pose.Pose(static_image_mode=False)
    
    ventana_camara = tk.Toplevel()
    ventana_camara.title("Procesando cámara")
    ventana_camara.geometry("800x600")
    
    lbl_video = tk.Label(ventana_camara)
    lbl_video.pack()
    def actualizar_frame():
        ret, frame = cap.read()
        if not ret:
            ventana_camara.after(10, actualizar_frame)
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame_rgb, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
            )

        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        lbl_video.imgtk = imgtk
        lbl_video.configure(image=imgtk)
        ventana_camara.after(10, actualizar_frame)

    actualizar_frame()

def interfaz():
    global combo_camaras

    ventana = tk.Tk()
    ventana.title("Ventana Principal")
    ventana.geometry("500x500")

    lbl_titulo = tk.Label(ventana, text="Selecciona el músculo a entrenar")
    lbl_titulo.pack(pady=20)

    btn_pierna = tk.Button(ventana, text="Pierna", command=ventanaPierna)
    btn_pierna.pack(pady=10)

    btn_pecho = tk.Button(ventana, text="Pecho", command=ventanaPecho)
    btn_pecho.pack(pady=10)

    btn_espalda = tk.Button(ventana, text="Espalda", command=ventanaEspalda)
    btn_espalda.pack(pady=10)

    lbl_combo = tk.Label(ventana, text="Selecciona una cámara")
    lbl_combo.pack(pady=10)

    camaras_disponibles = detectar_camaras()
    combo_camaras = ttk.Combobox(ventana, values=camaras_disponibles, state="readonly")
    if camaras_disponibles:
        combo_camaras.current(0)
    combo_camaras.pack(pady=10)

    btn_camara = tk.Button(ventana, text="Iniciar cámara y detectar poses", command=mostrar_camara)
    btn_camara.pack(pady=20)

    ventana.mainloop()
"""

import tkinter as tk
from tkinter import ttk
import customtkinter as ctk
from PIL import Image, ImageTk
import os
from Conexion import conectar_bd
import cv2
import mediapipe as mp
import numpy as np
from collections import Counter
from tensorflow.keras.models import load_model
from keras.models import load_model
import pickle
from Predecir import evaluar_sentadilla,convertir_landmarks_a_diccionario
from Predecir import main as predecirMain

model = load_model("modelo_ejercicios.h5")
with open("labels.pkl", "rb") as f:
    le = pickle.load(f)

def ventanaPierna():
    nueva_ventana = tk.Toplevel()
    nueva_ventana.title("Pierna")
    nueva_ventana.geometry("300x200")

def ventanaPecho():
    nueva_ventana = tk.Toplevel()
    nueva_ventana.title("Pecho")
    nueva_ventana.geometry("300x200")
    tk.Label(nueva_ventana, text="Esta es la ventana de pecho").pack(pady=20)

def ventanaEspalda():
    nueva_ventana = tk.Toplevel()
    nueva_ventana.title("Espalda")
    nueva_ventana.geometry("300x200")
    tk.Label(nueva_ventana, text="Esta es la ventana de espalda").pack(pady=20)

def detectar_camaras():
    indices = []
    for i in range(5):  # Escanear las primeras 5 cámaras
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            indices.append(str(i))
        cap.release()
    return indices

def mostrar_camara():
    global  marcas_error_global, nombre_ejercicio, zonaError
    zonaError = []  # Lista para almacenar las zonas de error
    nombre_ejercicio = "Desconocido"
    #zonaError = []  # Lista para almacenar las zonas de error
    marcas_error_global = []  # Lista para almacenar las marcas de error
    keypoints = []  # Lista para almacenar los keypoints

    cam_index = int(combo_camaras.get())
    cap = cv2.VideoCapture(cam_index)
    pose = mp.solutions.pose.Pose(static_image_mode=False)
    
    ventana_camara = tk.Toplevel()
    ventana_camara.title("Procesando cámara")
    ventana_camara.geometry("800x600")
    
    lbl_video = tk.Label(ventana_camara)
    lbl_video.pack()

    def actualizar_frame():
        global marcas_error_global,nombre_ejercicio,zonaError
        keypoints_cuerpo = []
        ventana_tamaño = 100  # Tamaño de la ventana de frames
        ret, frame = cap.read()
        if not ret:
            ventana_camara.after(10, actualizar_frame)
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            #mp.solutions.drawing_utils.draw_landmarks(
            #    frame_rgb, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
            #)
            # Obtener las coordenadas de la rodilla derecha
            altura, ancho, _ = frame.shape
            landmarks = results.pose_landmarks.landmark
            # Extraer keypoints de los landmarks    
            puntos = []
            for lm in results.pose_landmarks.landmark:
                puntos.extend([lm.x, lm.y, lm.z])
            
            keyCuerpo = convertir_landmarks_a_diccionario(results)
            keypoints_cuerpo.append(keyCuerpo)
            keypoints.append(puntos)

            if nombre_ejercicio == "squat":
                marcas_error_global = veredicto_squat(keypoints_cuerpo, landmarks, ancho, altura,zonaError)
                for marca in marcas_error_global:
                    if marca[0] == "punto":
                        _, x, y = marca
                        cv2.circle(frame_rgb, (x, y), 8, (0, 0, 255), -1)
                    elif marca[0] == "linea":
                        _, x1, y1, x2, y2 = marca
                        cv2.line(frame_rgb, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        # cv2.circle(frame_rgb, (rodilla_dx, rodilla_dy), 10, (255, 0, 0), -1)  # rojo en BGR
            
            if len(keypoints) == ventana_tamaño:
                nombre_ejercicio = predecir_ejercicio(keypoints)
                zonaError = evaluar_sentadilla(keypoints_cuerpo)
                print(f"Ejercicio detectado: {nombre_ejercicio}")
                # Reiniciar ventana para siguiente predicción
                keypoints.clear()
                keypoints_cuerpo.clear()
            

        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        lbl_video.imgtk = imgtk
        lbl_video.configure(image=imgtk)
        ventana_camara.after(10, actualizar_frame)

    actualizar_frame()

def veredicto_squat(keypoints_cuerpo, landmarks, ancho, altura,zonaError):
    marcas_error_global = []
    if len(keypoints_cuerpo) == 0:
        print("No se detectaron poses.")
        return "Desconocido"
    
    if not zonaError:
        print("No se detectaron errores en la sentadilla.")
        return []
    
    if "left_knee" in zonaError:
        rodilla_dx = int(landmarks[24].x * ancho)
        rodilla_dy = int(landmarks[24].y * altura)
        marcas_error_global.append(("punto", rodilla_dx,rodilla_dy ))
    if "right_hip" in zonaError:
        cadera_dx = int(landmarks[23].x * ancho)
        cadera_dy = int(landmarks[23].y * altura)
        marcas_error_global.append(("punto", cadera_dx, cadera_dy))
    if "back" in zonaError:
        hombro_dx = int(landmarks[12].x * ancho)
        hombro_dy = int(landmarks[12].y * altura)

        cadera_dx = int(landmarks[24].x * ancho)
        cadera_dy = int(landmarks[24].y * altura)
        marcas_error_global.append(("linea", cadera_dx, cadera_dy, hombro_dx, hombro_dy))
    
    return marcas_error_global

def predecir_ejercicio(keypoints):
    if len(keypoints) == 0:
        print("No se detectaron poses.")
        return "Desconocido"

    print("Extrayendo keypoints...")
    X = np.array(keypoints)

    print("Realizando predicciones...")
    preds = model.predict(X)
    clases_pred = np.argmax(preds, axis=1)

    clase_mayoritaria = Counter(clases_pred).most_common(1)[0][0]
    ejercicio = le.inverse_transform([clase_mayoritaria])[0]

    return ejercicio

def interfaz():
    global combo_camaras

    ventana = tk.Tk()
    ventana.title("Ventana Principal")
    ventana.geometry("500x500")

    lbl_titulo = tk.Label(ventana, text="Selecciona el músculo a entrenar")
    lbl_titulo.pack(pady=20)

    btn_pierna = tk.Button(ventana, text="Pierna", command=ventanaPierna)
    btn_pierna.pack(pady=10)

    btn_pecho = tk.Button(ventana, text="Pecho", command=ventanaPecho)
    btn_pecho.pack(pady=10)

    btn_espalda = tk.Button(ventana, text="Espalda", command=ventanaEspalda)
    btn_espalda.pack(pady=10)

    lbl_combo = tk.Label(ventana, text="Selecciona una cámara")
    lbl_combo.pack(pady=10)

    camaras_disponibles = detectar_camaras()
    combo_camaras = ttk.Combobox(ventana, values=camaras_disponibles, state="readonly")
    if camaras_disponibles:
        combo_camaras.current(0)
    combo_camaras.pack(pady=10)

    btn_camara = tk.Button(ventana, text="Iniciar cámara y detectar poses", command=mostrar_camara)
    btn_camara.pack(pady=20)

    ventana.mainloop()

def interfazInicial():
    global lbl_hora

    nombreUsuario = "Irvin"
    conexion = conectar_bd(nombreUsuario, "123")
    
    # Obtener la ruta absoluta del script
    ruta_script = os.path.dirname(os.path.abspath(__file__))

    ventana = ctk.CTk(fg_color="#FFFFFF")  # Fondo oscuro
    ventana.title("Interfaz Secundaria")
    ventana.geometry("1000x800")
    ventana.columnconfigure(0, weight=1)
    ventana.columnconfigure(1, weight=1)
    ventana.columnconfigure(2, weight=1)


    #Parte de arriba de la sesion
    frameSesion = ctk.CTkFrame(ventana, corner_radius=20, height=100,fg_color="#c4d2f4")
    frameSesion.columnconfigure(0, weight=1)
    frameSesion.columnconfigure(1, weight=1)
    frameSesion.columnconfigure(2, weight=1)
    frameSesion.grid(row=0, column=0, columnspan=3,sticky="nsew", padx=20, pady=20)

    imagen = Image.open(os.path.join(ruta_script, "..", "Imagenes", "loginWorkout.png"))
    lbl_fotoSesion = ctk.CTkLabel(frameSesion, image=ImageTk.PhotoImage(imagen), text="")
    lbl_fotoSesion.grid(row=0, column=0, padx=10, pady=10, sticky="w")

    lbl_datosSesion = ctk.CTkLabel(frameSesion, text=f"Hola: {nombreUsuario} ", font=("Arial", 14, "bold"), fg_color="#c4d2f4", text_color="#393E46")
    lbl_datosSesion.grid(row=0, column=0, padx = (55,0),pady=10, sticky="w")

    imagen = Image.open(os.path.join(ruta_script, "..", "Imagenes", "campanaWorkout.png"))
    imagen_tk = ImageTk.PhotoImage(imagen)
    btn_notificaciones = ctk.CTkButton(frameSesion,image=imagen_tk,text="",corner_radius=20,width=30,fg_color="#9eb8f9")
    btn_notificaciones.grid(row=0, column=2, padx=10, pady=10, sticky="e")


    # Parte central de la ventana
    imagen = Image.open(os.path.join(ruta_script, "..", "Imagenes", "logoWorkout.png"))
    imagen_tk = ImageTk.PhotoImage(imagen)
    lbl_espacio = ctk.CTkLabel(ventana,image=imagen_tk, text="",width=184,height=184,fg_color="#ffffff")
    lbl_espacio.image = imagen_tk  # Mantener una referencia a la imagen
    lbl_espacio.grid(row=1, column=1, padx=10, pady=10)

    frameTimer = ctk.CTkFrame(ventana, corner_radius=20, width=250, height=225,fg_color="#c4d2f4")
    frameTimer.columnconfigure(0, weight=1)
    frameTimer.columnconfigure(1, weight=1)
    frameTimer.columnconfigure(2, weight=1)
    frameTimer.grid_propagate(False)
    frameTimer.grid(row=3, column=0, padx=20, pady=20)

    lbl_timer = tk.Label(frameTimer, text="Tiempo", font=("Arial", 14, "bold"), bg="#c4d2f4", fg="#393E46")
    lbl_timer.grid(row=0, column=0, padx = 30,pady=(20,30),sticky="w")

    imagen = Image.open(os.path.join(ruta_script, "..", "Imagenes", "pngwing.com.png"))
    imagen_tk = ImageTk.PhotoImage(imagen)
    lbl_imagen = tk.Label(frameTimer, image=imagen_tk, bg="#c4d2f4",height=20, width=20)
    lbl_imagen.image = imagen_tk  # Mantener una referencia a la imagen
    lbl_imagen.grid(row=0, column=1, padx=10, pady=(20,30), sticky="e")

    lbl_fecha = tk.Label(frameTimer, text= mostrar_fecha(), font=("Arial", 12, "bold"), bg="#c4d2f4", fg="#393E46")
    lbl_fecha.grid(row=1, column=0, padx = 30,pady=20,sticky="w")

    imgaenCalendario = Image.open(os.path.join(ruta_script, "..", "Imagenes", "calendarioWorkout.png"))
    imagen_tkCalendario = ImageTk.PhotoImage(imgaenCalendario)
    lbl_imagenCalendario = tk.Label(frameTimer, image=imagen_tkCalendario, bg="#c4d2f4",height=20, width=20)
    lbl_imagenCalendario.image = imagen_tkCalendario  # Mantener una referencia a la imagen
    lbl_imagenCalendario.grid(row=1, column=1, padx=10, pady=(20,5), sticky="e")
    
    lbl_hora = tk.Label(frameTimer, text= "", font=("Arial", 18, "bold"),bg="#c4d2f4", fg="#393E46")
    lbl_hora.grid(row=2, column=0, padx = 30,pady=(20,5),sticky="w")
    
    lbl_horaTexto = tk.Label(frameTimer, text="Horas", font=("Arial", 12, "bold"), bg="#c4d2f4", fg="#393E46")
    lbl_horaTexto.grid(row=3, column=0, padx = 30,sticky="w")

    """ventana,
    text="Iniciar cámara",
    font=("Arial", 14, "bold"),
    bg="#222831",        # Fondo oscuro (como transparente en tema oscuro)
    fg="#EEEEEE",        # Texto claro
    activebackground="#393E46",  # Color cuando haces clic
    activeforeground="#00ADB5",  # Color del texto cuando haces clic
    borderwidth=0,       # Sin borde
    width=25,
    height=
    )"""

    # Botón Iniciar Cámara (centro)
    btn_camara =ctk.CTkButton(
    ventana,
    text="Iniciar cámara",
    font=("Arial", 18, "bold"),
    corner_radius=20,  # ¡Esto sí redondea!
    fg_color="#00ADB5",
    text_color="white",
    width=250,
    height=75,
    command=interfaz  # Llama a la función mostrar_camara al hacer clic
    )   
    btn_camara.grid(row=2, column=2, padx=10, pady=10)

    # Botón Entrenar Modelo (centro)
    btn_entrenar = ctk.CTkButton(
    ventana,
    text="Entrenar modelo",
    font=("Arial", 18, "bold"),
    corner_radius=20,  # ¡Esto sí redondea!
    fg_color="#00ADB5",
    text_color="white",
    width=250,
    height=75
    )   
    btn_entrenar.grid(row=2, column=0, pady=10)

    # Botón Video Grabado (centro)
    btn_videoGuardado = ctk.CTkButton(
    ventana,
    text="Ingresar video grabado",
    font=("Arial", 18, "bold"),
    corner_radius=20,  # ¡Esto sí redondea!
    fg_color="#00ADB5",
    text_color="white",
    width=250,
    height=75,
    command=predecirMain
    )   
    btn_videoGuardado.grid(row=2, column=1, pady=10)

    # Botón Cerrar (abajo izquierda)
    btn_cerrar = ctk.CTkButton(
    ventana,
    text="Cerrar",
    font=("Arial", 18, "bold"),
    corner_radius=20,  # ¡Esto sí redondea!
    fg_color="#00ADB5",
    text_color="white",
    width=25,
    height=10
    )   
    btn_cerrar.grid(row=4, column=0, padx=10, pady=50, sticky="w")

    mostrar_hora()  # Iniciar la actualización de la hora
    ventana.mainloop()

def mostrar_fecha():
    from datetime import datetime
    fecha_actual = datetime.now().strftime("%Y-%m-%d")
    return fecha_actual

def mostrar_hora():
    global lbl_hora
    from datetime import datetime
    hora_actual = datetime.now().strftime("%H:%M:%S")
    lbl_hora.configure(text=hora_actual)
    # Llama esta función otra vez después de 1000 ms (1 segundo)
    lbl_hora.after(1000, mostrar_hora)

def main():
    interfazInicial()

if __name__ == "__main__":
    main()