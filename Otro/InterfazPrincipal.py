import threading
import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # '0' = all logs, '1' = filter INFO, '2' = filter WARNING, '3' = only ERROR
from Conexion import conectar_bd
import cv2
import mediapipe as mp
import numpy as np
from collections import Counter
from keras.models import load_model
import pickle
from Predecir import convertir_landmarks_a_diccionario
from Proyecto import main as entrenarIA
from EvaluarEjericios import *
from Conexion import *
from pygrabber.dshow_graph import FilterGraph
import os
from VentanasSecundarias import interfazInicioSesion, interfaz_mostrar_estadisticas, interfaz_mostrar_notificaciones, interfaz_subir_video
from Utilidades import  centrar_ventana, cargar_credenciales
from UtilEjercicio import evaluarEjercicio, predecir_ejercicio

# Variables globales
model = None
le = None
modelo_listo = False

conexion = conectar_bd()
nombreUsuario = "Usuario Desconocido"
password = ""
usuarioID = None

def cargar_modelo_y_labels():
    global model, le, modelo_listo
    print("Cargando modelo y etiquetas...")
    model = load_model("modelo_ejercicios.h5")
    with open("labels.pkl", "rb") as f:
        le = pickle.load(f)
    modelo_listo = True
    print("Modelo y etiquetas cargados.")

def detectar_camaras():
    camaras_disponibles = []
    graph = FilterGraph()
    dispositivos = graph.get_input_devices()

    for i, nombre in enumerate(dispositivos):
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            camaras_disponibles.append((str(i), nombre))
        cap.release()

    return camaras_disponibles

def mostrar_camara():
    global nombre_ejercicio, zonaError, animar, repeticiones, keypoints_cuerpo,estadisticas, btn_camara
    keypoints_cuerpo = []
    repeticiones = 0
    zonaError = []  # Lista para almacenar las zonas de error
    estadisticas = []  # Lista para almacenar las estadísticas
    nombre_ejercicio = "Desconocido"
    keypoints = []  # Lista para almacenar los keypoints

    cam_index = int(opcion.get())  # Obtener el índice de la cámara seleccionada
    cap = cv2.VideoCapture(cam_index)
    pose = mp.solutions.pose.Pose(static_image_mode=False)

    #animar = False
    ventana_camara = tk.Toplevel()
    ventana_camara.title("Procesando cámara")
    ventana_camara.geometry("800x600")

    lbl_video = tk.Label(ventana_camara)
    lbl_video.pack()
    def cerrar_camara():
        # Actualiza las estadísticas de todos los ejercicios realizados en la sesión
        for est in estadisticas:
            actualizar_estadisticas(conexion, usuarioID, est["nombre"], est["repeticiones"], len(zonaError), 10)
        cap.release()
        ventana_camara.destroy()

    btn_cerrar = tk.Button(ventana_camara, text="Cerrar cámara", command=cerrar_camara, bg="#00ADB5", fg="white", font=("Arial", 12, "bold"))
    btn_cerrar.pack(pady=10)

    detener_animacion(btn_camara)

    def actualizar_frame():
        global nombre_ejercicio, zonaError, frame_rgb, repeticiones, lbl_repeticiones, keypoints_cuerpo, estadisticas
        ventana_tamaño = 100  # Tamaño de la ventana de frames
        ret, frame = cap.read()
        if not ret:
            ventana_camara.after(10, actualizar_frame)
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            altura, ancho, _ = frame.shape
            landmarks = results.pose_landmarks.landmark
            puntos = []
            for lm in results.pose_landmarks.landmark:
                puntos.extend([lm.x, lm.y, lm.z])

            keyCuerpo = convertir_landmarks_a_diccionario(results)
            keypoints_cuerpo.append(keyCuerpo)
            keypoints.append(puntos)

            evaluarEjercicio(nombre_ejercicio, keypoints_cuerpo, landmarks, ancho, altura, zonaError, frame_rgb)

            if len(keypoints) == ventana_tamaño:
                zonaError = []
                repeticionesAux = 0
                nombre_ejercicio = predecir_ejercicio(keypoints,model, le)
                match nombre_ejercicio:
                    case "squat":
                        zonaError, repeticionesAux = evaluar_sentadilla(keypoints_cuerpo)
                    case "barbell biceps curl":
                        zonaError, repeticionesAux = evaluar_curl_biceps(keypoints_cuerpo)
                    case "pull up":
                        zonaError, repeticionesAux = evaluar_pullup(keypoints_cuerpo)
                    case _:
                        zonaError = []
                repeticiones += repeticionesAux
                lbl_repeticiones.configure(text=f"Repeticiones: {repeticiones}")
                print(f"Ejercicio detectado: {nombre_ejercicio}")
                # Actualizar la lista de estadísticas
                # Buscar si ya existe el ejercicio en la lista
                encontrado = False
                for est in estadisticas:
                    if est["nombre"] == nombre_ejercicio:
                        est["repeticiones"] += repeticionesAux
                        encontrado = True
                        break
                if not encontrado and nombre_ejercicio != "Desconocido":
                    estadisticas.append({"nombre": nombre_ejercicio, "repeticiones": repeticionesAux})

                keypoints.clear()
                keypoints_cuerpo.clear()

        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        lbl_video.imgtk = imgtk
        lbl_video.configure(image=imgtk)
        ventana_camara.after(10, actualizar_frame)

    actualizar_frame()


def cargarInterfazCamaras(btn):
    global camaras_disponibles,ventanaInicial
    camaras_disponibles = []

    def cargar_camaras():
        global camaras_disponibles,frameCamaras,opcion,animar
        camaras_disponibles = detectar_camaras()
        for i, (indice, nombre) in enumerate(camaras_disponibles):
            camaras_disponibles[i] = f"{indice} - {nombre}"
            # Crear los radio buttons
            radio = ctk.CTkRadioButton(frameCamaras, text=f"{nombre}",
                                        variable=opcion, 
                                        value=f"{i}",
                                        fg_color="#1F3E3E", 
                                        text_color="#393E46",
                                        border_color="#FFFFFF")
            radio.grid(row=i + 2, column=0, padx=10, pady=10, sticky="w")
        btn.animar = False

    animar_texto(btn)
    threading.Thread(target=cargar_camaras).start()



def animar_texto(btn):
    # Usa un atributo en el botón para controlar la animación individual
    btn.animar = True
    textoAnterior = btn.cget("text")
    def ciclo(i=0):
        if getattr(btn, "animar", False):
            btn.configure(state="disabled")
            puntos = "." * (i % 4)
            btn.configure(text=f"Cargando{puntos}")
            ventanaInicial.after(300, ciclo, i + 1)    
        else:
            btn.configure(text=textoAnterior)
            btn.configure(state="normal")
    ciclo()

def detener_animacion(btn):
    btn.animar = False

def cargarCamara(btn):
    global lbl_txtSeleccion
    if opcion.get() == "":
        lbl_txtSeleccion.configure(text_color = "#ff0000")
    else:
        lbl_txtSeleccion.configure(text_color = "#393E46")
        animar_texto(btn)
        threading.Thread(target = mostrar_camara).start()

def interfazInicial():
    global lbl_hora,ventanaInicial, frameCamaras, opcion,nombreUsuario

    # Obtener la ruta absoluta del script
    ruta_script = os.path.dirname(os.path.abspath(__file__))

    ventanaInicial = ctk.CTk(fg_color="#000000")  # Fondo oscuro
    ventanaInicial.title("Interfaz Secundaria")

    centrar_ventana(ventanaInicial, 1000, 800)  # Centrar ventana
    ventanaInicial.columnconfigure(0, weight=1)
    ventanaInicial.columnconfigure(1, weight=1)
    ventanaInicial.columnconfigure(2, weight=1)
    #ventanaInicial.overrideredirect(True)  # Elimina la barra de título

    #Parte de arriba de la sesion
    frameSesion = ctk.CTkFrame(ventanaInicial, corner_radius=20, height=100,fg_color="#c4d2f4")
    frameSesion.columnconfigure(0, weight=1)
    frameSesion.columnconfigure(1, weight=1)
    frameSesion.columnconfigure(2, weight=1)
    frameSesion.grid(row=0, column=0, columnspan=3,sticky="nsew", padx=20, pady=20)

    imagen = Image.open(os.path.join(ruta_script, "..", "Imagenes", "loginWorkout.png"))
    imagen_tk_login = ImageTk.PhotoImage(imagen)
    btn_fotoSesion = ctk.CTkButton(frameSesion, image=imagen_tk_login, 
                                    text="", width=40, height=40,
                                    fg_color="transparent",
                                    hover_color="#a1b9ed",
                                    command=lambda: interfazInicioSesion())
    btn_fotoSesion.image = imagen_tk_login  # Mantener referencia
    btn_fotoSesion.grid(row=0, column=0, padx=10, pady=10, sticky="w")



    lbl_datosSesion = ctk.CTkLabel(frameSesion, text=f"Hola, {nombreUsuario} ", font=("Arial", 14, "bold"), fg_color="#c4d2f4", text_color="#393E46")
    lbl_datosSesion.grid(row=0, column=0, padx = (65,0),pady=10, sticky="w")

    imagen = Image.open(os.path.join(ruta_script, "..", "Imagenes", "campanaWorkout.png"))
    imagen_tk = ImageTk.PhotoImage(imagen)
    btn_notificaciones = ctk.CTkButton(frameSesion,image=imagen_tk,text="",corner_radius=20,width=30,fg_color="#9eb8f9",command=lambda: interfaz_mostrar_notificaciones())
    btn_notificaciones.grid(row=0, column=2, padx=10, pady=10, sticky="e")

    frameLogo = ctk.CTkFrame(ventanaInicial, corner_radius=20, height=180)
    frameLogo.columnconfigure(0, weight=1)
    frameLogo.columnconfigure(1, weight=1)
    frameLogo.columnconfigure(2, weight=1)
    #frameLogo.grid_propagate(False)
    frameLogo.grid(row=1, column=0,columnspan=3, sticky="nsew", padx=20, pady=20)

    fondo = Image.open(os.path.join(ruta_script, "..", "Imagenes", "fondoWorkout.png")).convert("RGBA")
    logo = Image.open(os.path.join(ruta_script, "..", "Imagenes", "logoWorkout.png")).convert("RGBA")

    x = (fondo.width - logo.width) // 2
    y = (fondo.height - logo.height) // 2

    fondo.paste(logo, (x, y), logo)
    imagen_combinada = ImageTk.PhotoImage(fondo)

    #imagen_tk = ImageTk.PhotoImage(imagen)
    lbl_fondo = ctk.CTkLabel(frameLogo, image= imagen_combinada, text="",height=180)
    lbl_fondo.image = imagen_tk  # Mantener una referencia a la imagen
    lbl_fondo.place(x=0, y=0, relwidth=1, relheight=1)

    frameTimer = ctk.CTkFrame(ventanaInicial, corner_radius=20, width=250, height=225,fg_color="#c4d2f4")
    frameTimer.columnconfigure(0, weight=1)
    frameTimer.columnconfigure(1, weight=1)
    frameTimer.columnconfigure(2, weight=1)
    frameTimer.grid_propagate(False)
    frameTimer.grid(row=3, column=0, padx=20, pady=20)

    lbl_timer = tk.Label(frameTimer, text="Tiempo", font=("Arial", 14, "bold"), bg="#c4d2f4", fg="#393E46")
    lbl_timer.grid(row=0, column=0, padx = 30,pady=(20,30),sticky="w")

    imagen = Image.open(os.path.join(ruta_script, "..", "Imagenes", "pngwing.com.png"))
    imagen_tk = ImageTk.PhotoImage(imagen)
    lbl_imagen = tk.Label(frameTimer, image=imagen_tk, bg="#c4d2f4",height=20, width=20,)
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

    frameEstadisticas = ctk.CTkFrame(ventanaInicial, corner_radius=20, width=250, height=225,fg_color="#c4d2f4")
    frameEstadisticas.columnconfigure(0, weight = 1)
    frameEstadisticas.grid_propagate(False)
    frameEstadisticas.grid(row = 3, column = 1,padx = 10,pady = 20)

    btn_estadisticas = ctk.CTkButton(frameEstadisticas,fg_color="transparent",
                                        text="Estadisticas",
                                        font=("Arial", 18, "bold"),
                                        corner_radius=20,  # ¡Esto sí redondea!
                                        text_color="black",)
    btn_estadisticas.grid(row = 0,column = 0,padx = 10,pady = (10,0))
    btn_estadisticas.configure(command = lambda: interfaz_mostrar_estadisticas(usuarioID))

    imagen_estadisticas = Image.open(os.path.join(ruta_script, "..", "Imagenes", "estadisticasWorkout.png"))
    imagen_tk_estadisticas = ImageTk.PhotoImage(imagen_estadisticas)
    lbl_imagen_estadisticas = ctk.CTkLabel(frameEstadisticas, image=imagen_tk_estadisticas, text="")
    lbl_imagen_estadisticas.image = imagen_tk_estadisticas  # Mantener una referencia a la imagen
    lbl_imagen_estadisticas.grid(row=1, column=0, padx=10)

    global lbl_repeticiones
    lbl_repeticiones = ctk.CTkLabel(frameEstadisticas, text="Repeticiones: 0 ", font=("Arial", 14, "bold"), fg_color="#c4d2f4", text_color="#393E46")
    #lbl_repeticiones.grid(row = 2,column = 0,padx=20, pady=(0, 10), sticky="w")

    frameCamaras = ctk.CTkFrame(ventanaInicial, corner_radius=20, width=250, height=225,fg_color="#c4d2f4")
    frameCamaras.columnconfigure(0, weight=1)
    frameCamaras.grid_propagate(False)
    frameCamaras.grid(row = 3, column = 2, padx=20, pady=20)

    lbl_txtCamaras = ctk.CTkLabel(frameCamaras, text="Cámaras disponibles: ", font=("Arial", 14, "bold"), fg_color="#c4d2f4", text_color="#393E46")
    lbl_txtCamaras.grid(row=0, column=0, padx=10, pady=10)
    
    global lbl_txtSeleccion
    lbl_txtSeleccion = ctk.CTkLabel(frameCamaras, text="Selecciona una cámara", font=("Arial", 14, "bold"), fg_color="#c4d2f4", text_color="#393E46")
    lbl_txtSeleccion.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="w")

    opcion = ctk.StringVar(value="")  
    global btn_camara
    # Botón Iniciar Cámara (centro)
    btn_camara =ctk.CTkButton(
    ventanaInicial,
    text="Iniciar cámara",
    font=("Arial", 18, "bold"),
    corner_radius=20,  # ¡Esto sí redondea!
    fg_color="#00ADB5",
    text_color="white",
    width=250,
    height=75
    )   
    btn_camara.grid(row=2, column=2, padx=10, pady=10)
    btn_camara.configure(command = lambda: cargarCamara(btn_camara))
    threading.Thread(target=cargarInterfazCamaras(btn_camara)).start() 

    # Botón Entrenar Modelo (centro)
    btn_entrenar = ctk.CTkButton(
    ventanaInicial,
    text="Entrenar modelo",
    font=("Arial", 18, "bold"),
    corner_radius=20,  # ¡Esto sí redondea!
    fg_color="#00ADB5",
    text_color="white",
    width=250,
    height=75,
    command=lambda: entrenarIA()# Llama a la función entrenarIA al hacer clic
    )   
    btn_entrenar.grid(row=2, column=0, pady=10)



    # Botón para predecir video en un hilo
    btn_predecir_video = ctk.CTkButton(
        ventanaInicial,
        text="Predecir video",
        font=("Arial", 18, "bold"),
        corner_radius=20,
        fg_color="#00ADB5",
        text_color="white",
        width=250,
        height=75,
        command=lambda: threading.Thread(target=interfaz_subir_video(model,le,modelo_listo)).start()
    )
    btn_predecir_video.grid(row=2, column=1, pady=10)


    # Botón Cerrar (abajo izquierda)
    btn_cerrar = ctk.CTkButton(
    ventanaInicial,
    text="Cerrar",
    font=("Arial", 18, "bold"),
    corner_radius=20,  # ¡Esto sí redondea!
    fg_color="#00ADB5",
    text_color="white",
    width=25,
    height=10,
    command=ventanaInicial.destroy  # Cierra la ventana al hacer clic
    )   
    btn_cerrar.grid(row=4, column=0, padx=10, pady=50, sticky="w")

    mostrar_hora()  # Iniciar la actualización de la hora
    ventanaInicial.mainloop()

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
    global nombreUsuario, password, usuarioID
    threading.Thread(target=cargar_modelo_y_labels).start()

    nombreUsuario, password, usuarioID = cargar_credenciales()

    interfazInicial()


if __name__ == "__main__":
    main()