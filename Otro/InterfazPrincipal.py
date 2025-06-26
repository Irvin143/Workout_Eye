import threading
import tkinter as tk
from tkinter import ttk
import customtkinter as ctk
from PIL import Image, ImageTk
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # '0' = all logs, '1' = filter INFO, '2' = filter WARNING, '3' = only ERROR
from Conexion import conectar_bd
import cv2
import mediapipe as mp
import numpy as np
from collections import Counter
from tensorflow.keras.models import load_model
from keras.models import load_model
import pickle
from Predecir import convertir_landmarks_a_diccionario
from Predecir import main as predecirMain
from EvaluarEjericios import *
from Conexion import *
from pygrabber.dshow_graph import FilterGraph
import json
import os

# Variables globales
model = None
le = None
modelo_listo = False

conexion = conectar_bd()
nombreUsuario = "Usuario Desconocido"
password = ""
usuarioID = None
# filepath: c:\VisualStudio\Python\WorkoutEye\Otro\InterfazPrincipal.py
# ...existing code...

CONFIG_FILE = "inicioSesion.json"
def guardar_credenciales(usuario, contra, usuarioid):
    with open(CONFIG_FILE, "w") as f:
        json.dump({"usuario": usuario, "contra": contra, "usuarioID": usuarioid}, f)

def cargar_credenciales():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            data = json.load(f)
            return data.get("usuario"), data.get("contra"),data.get("usuarioID")
    return None, None, None


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
    global   nombre_ejercicio, zonaError,animar,repeticiones,keypoints_cuerpo
    keypoints_cuerpo = []
    repeticiones = 0
    zonaError = []  # Lista para almacenar las zonas de error
    nombre_ejercicio = "Desconocido"
    keypoints = []  # Lista para almacenar los keypoints

    cam_index = int(opcion.get())  # Obtener el índice de la cámara seleccionada
    cap = cv2.VideoCapture(cam_index)
    pose = mp.solutions.pose.Pose(static_image_mode = False)
    
    animar = False
    ventana_camara = tk.Toplevel()
    ventana_camara.title("Procesando cámara")
    ventana_camara.geometry("800x600")
    
    lbl_video = tk.Label(ventana_camara)
    lbl_video.pack()

    def actualizar_frame():
        global nombre_ejercicio,zonaError, frame_rgb,repeticiones,lbl_repeticiones,keypoints_cuerpo
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
            # Keypoints_cuerpo es un diccionario con las coordenadas de los landmarks
            keypoints_cuerpo.append(keyCuerpo)
            #keypoints son los puntos en formato lista
            keypoints.append(puntos)

            evaluarEjercicio(nombre_ejercicio, keypoints_cuerpo, landmarks, ancho, altura,zonaError)

            if len(keypoints) == ventana_tamaño:
                zonaError = []
                nombre_ejercicio = predecir_ejercicio(keypoints)
                match nombre_ejercicio:
                    case "squat":
                        zonaError = evaluar_sentadilla(keypoints_cuerpo)
                    case "barbell biceps curl":
                        repeticionesAux = 0
                        zonaError, repeticionesAux = evaluar_curl_biceps(keypoints_cuerpo)
                        repeticiones += repeticionesAux
                        lbl_repeticiones.configure(text = f"Repeticiones: {repeticiones}")
                    case _:
                        zonaError = []
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

def evaluarEjercicio(nombreEjercicio, keypoints_cuerpo, landmarks, ancho, altura,zonaError):
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
        animar = False

    animar_texto(btn)
    threading.Thread(target=cargar_camaras).start()

def animar_texto(btn):
    global animar 
    animar = True
    textoAnterior = btn.cget("text")
    def ciclo(i=0):
        if animar:
            btn.configure(state="disabled")
            puntos = "." * (i % 4)
            btn.configure(text=f"Cargando{puntos}")
            ventanaInicial.after(300, ciclo, i + 1)
        else:
            btn.configure(text=textoAnterior)
            btn.configure(state="normal")
    ciclo()

def cargarCamara(btn):
    global lbl_txtSeleccion
    if opcion.get() == "":
        lbl_txtSeleccion.configure(text_color = "#ff0000")
    else:
        lbl_txtSeleccion.configure(text_color = "#393E46")
        animar_texto(btn)
        threading.Thread(target = mostrar_camara).start()

def centrar_ventana(ventana, ancho, alto):
    ventana.update_idletasks()  # Asegura que obtenga tamaño de pantalla real
    screen_width = ventana.winfo_screenwidth()
    screen_height = ventana.winfo_screenheight()

    x = (screen_width // 2) - (ancho // 2)
    y = (screen_height // 2) - (alto // 2)

    ventana.geometry(f"{ancho}x{alto}+{x}+{y}")

def interfazInicial():
    global lbl_hora,ventanaInicial, frameCamaras, opcion,nombreUsuario

    # Obtener la ruta absoluta del script
    ruta_script = os.path.dirname(os.path.abspath(__file__))

    ventanaInicial = ctk.CTk(fg_color="#FFFFFF")  # Fondo oscuro
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
    btn_notificaciones = ctk.CTkButton(frameSesion,image=imagen_tk,text="",corner_radius=20,width=30,fg_color="#9eb8f9")
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
    btn_estadisticas.configure(command = mostrar_estadisticas)

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
    height=75
    )   
    btn_entrenar.grid(row=2, column=0, pady=10)

    # Botón Video Grabado (centro)
    btn_videoGuardado = ctk.CTkButton(
    ventanaInicial,
    text="Ingresar video grabado",
    font=("Arial", 18, "bold"),
    corner_radius=20,  # ¡Esto sí redondea!
    fg_color="#00ADB5",
    text_color="white",
    width=250,
    height=75,
    command=lambda: predecirMain(model, le)  # Llama a la función predecirMain al hacer clic
    )   
    btn_videoGuardado.grid(row=2, column=1, pady=10)

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
    import time
    hora_actual = datetime.now().strftime("%H:%M:%S")
    lbl_hora.configure(text=hora_actual)
    # Llama esta función otra vez después de 1000 ms (1 segundo)
    lbl_hora.after(1000, mostrar_hora)

def interfazInicioSesion():
    # Crear ventana de inicio de sesión
    ventana_login = ctk.CTk()
    ventana_login.title("Inicio de Sesión")
    ventana_login.geometry("500x400")
    ventana_login.resizable(False, False)
    centrar_ventana(ventana_login, 500, 400)
    ventana_login.configure(fg_color="#c4d2f4")

    # Fondo decorativo
    ruta_script = os.path.dirname(os.path.abspath(__file__))
    try:
        fondo = Image.open(os.path.join(ruta_script, "..", "Imagenes", "fondoWorkout.png")).convert("RGBA")
        fondo = fondo.resize((500, 400))
        fondo_img = ImageTk.PhotoImage(fondo)
        lbl_fondo = tk.Label(ventana_login, image=fondo_img, borderwidth=0)
        lbl_fondo.place(x=0, y=0, relwidth=1, relheight=1)
        lbl_fondo.image = fondo_img
    except Exception:
        pass

    # Marco para el formulario
    frame_form = ctk.CTkFrame(ventana_login, corner_radius=20, fg_color="#ffffff", width=350, height=300)
    frame_form.place(relx=0.5, rely=0.5, anchor="center")

    # Logo
    try:
        logo = Image.open(os.path.join(ruta_script, "..", "Imagenes", "logoWorkout.png")).convert("RGBA")
        logo = logo.resize((80, 80))
        logo_img = ImageTk.PhotoImage(logo)
        lbl_logo = tk.Label(frame_form, image=logo_img, bg="#ffffff", borderwidth=0)
        lbl_logo.image = logo_img
        lbl_logo.pack(pady=(20, 10))
    except Exception:
        lbl_logo = tk.Label(frame_form, text="WorkoutEye", font=("Arial", 20, "bold"), bg="#ffffff", fg="#393E46")
        lbl_logo.pack(pady=(20, 10))

    # Etiqueta de bienvenida
    lbl_bienvenida = ctk.CTkLabel(frame_form, text="Bienvenido", font=("Arial", 18, "bold"), fg_color="#ffffff", text_color="#393E46")
    lbl_bienvenida.pack(pady=(0, 10))
    # Campo usuario
    entry_usuario = ctk.CTkEntry(frame_form, placeholder_text="Usuario", width=220, height=35, corner_radius=10, fg_color="#e6eaf8", text_color="#393E46")
    entry_usuario.pack(padx = 15,pady=10)
    # Dar el foco automáticamente al campo de usuario al abrir la ventana
    ventana_login.after(100, entry_usuario.focus_set)

    # Campo contraseña
    entry_contra = ctk.CTkEntry(frame_form, placeholder_text="Contraseña", show="*", width=220, height=35, corner_radius=10, fg_color="#e6eaf8", text_color="#393E46")
    entry_contra.pack(padx = 15,pady=10)

    # Etiqueta de error
    lbl_error = ctk.CTkLabel(frame_form, text="", font=("Arial", 12), fg_color="#ffffff", text_color="#ff0000")
    lbl_error.pack()

    # Función de login
    def login(event=None):  # acepta event para el bind
        usuarioid = None
        usuario = entry_usuario.get()
        contra = entry_contra.get()
        if usuario and contra:
            usuarioid = grabarUsuario(conexion, usuario, contra, "M", 0)  # Genero y edad por defecto
            print(f"Usuario insertado con ID: {usuarioid}")
            if usuarioid is not None:
                guardar_credenciales(usuario, contra,usuarioid)
                ventana_login.destroy()
                interfazInicial()
            else:
                lbl_error.configure(text="Usuario o contraseña incorrectos")
        else:
            lbl_error.configure(text="Completa todos los campos")

    # Botón de inicio de sesión
    btn_login = ctk.CTkButton(frame_form, text="Iniciar Sesión", fg_color="#00ADB5", text_color="white", font=("Arial", 14, "bold"),
                                width=180, height=40, corner_radius=10, command=login)
    btn_login.pack(pady=(10, 10))

    # Vincular Enter a la función de login
    ventana_login.bind('<Return>', login)

    # Pie de página
    lbl_footer = ctk.CTkLabel(frame_form, text="© 2024 WorkoutEye", font=("Arial", 10), fg_color="#ffffff", text_color="#393E46")
    lbl_footer.pack(side="bottom", pady=(10, 5))
    
    ventana_login.mainloop()

def mostrar_estadisticas():
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import random
    from datetime import datetime, timedelta

    # Ventana de estadísticas a pantalla completa
    global conexion, usuarioID
    diccEstadisticas = []
    historial = []
    repeticiones, erroresPostura, puntajeTecnica,nombreEjercicio = [],[],[],[]
    diccEstadisticas = consultar_estadisticas(conexion, usuarioID)

    for estadistica in diccEstadisticas:
        repeticiones.append(int(estadistica['Repeticiones']))
        erroresPostura.append(estadistica['ErroresPostura'])
        puntajeTecnica.append(estadistica['PuntajeTecnica'])
        nombreEjercicio.append(estadistica['NombreEjercicio'])
        historial.append((
            "24-06-2025",  # fecha vacía
            estadistica['NombreEjercicio'],
            int(estadistica['Repeticiones'])
        ))

    ventana_stats = ctk.CTkToplevel()
    ventana_stats.title("Estadísticas de Ejercicios")
    # Maximiza la ventana pero sin quitar la barra de título ni poner fullscreen real
    ventana_stats.state('zoomed')  # Para Windows: maximiza la ventana
    # ventana_stats.attributes('-fullscreen', True)  # No usar fullscreen real para no ocultar barra de título
    ventana_stats.configure(fg_color="#c4d2f4")

    # Botón para salir/cerrar estadísticas
    def salir_stats():
        ventana_stats.destroy()

    # Frame principal ocupa toda la ventana
    frame_main = ctk.CTkFrame(ventana_stats, fg_color="#ffffff", corner_radius=15)
    frame_main.pack(fill="both", expand=True, padx=40, pady=(40, 20))
    frame_main.grid_rowconfigure(0, weight=1)
    frame_main.grid_rowconfigure(1, weight=0)
    frame_main.grid_rowconfigure(2, weight=0)
    frame_main.grid_rowconfigure(3, weight=0)
    frame_main.grid_columnconfigure(0, weight=2)
    frame_main.grid_columnconfigure(1, weight=1)

    # Frame para la gráfica (izquierda arriba)
    frame_grafica = ctk.CTkFrame(frame_main, fg_color="#ffffff", corner_radius=10)
    frame_grafica.grid(row=0, column=0, sticky="nsew", padx=(10, 10), pady=(10, 5))

    # Frame para detalles (derecha arriba)
    frame_detalles = ctk.CTkFrame(frame_main, fg_color="#e6eaf8", corner_radius=10)
    frame_detalles.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=(10, 5))

    # Frame para resumen (derecha medio)
    frame_resumen = ctk.CTkFrame(frame_main, fg_color="#e6eaf8", corner_radius=10)
    frame_resumen.grid(row=1, column=1, sticky="nsew", padx=(0, 10), pady=(5, 5))

    # Frame para progreso semanal (derecha abajo)
    frame_progreso = ctk.CTkFrame(frame_main, fg_color="#e6eaf8", corner_radius=10)
    frame_progreso.grid(row=2, column=1, sticky="nsew", padx=(0, 10), pady=(5, 5))

    # Frame para historial (izquierda medio)
    frame_historial = ctk.CTkFrame(frame_main, fg_color="#e6eaf8", corner_radius=10)
    frame_historial.grid(row=1, column=0, rowspan=2, sticky="nsew", padx=(10, 10), pady=(5, 5))

    # Frame para ranking (izquierda abajo)
    frame_ranking = ctk.CTkFrame(frame_main, fg_color="#e6eaf8", corner_radius=10)
    frame_ranking.grid(row=3, column=0, sticky="nsew", padx=(10, 10), pady=(5, 10))

    # Frame para consejos (derecha abajo)
    frame_consejos = ctk.CTkFrame(frame_main, fg_color="#e6eaf8", corner_radius=10)
    frame_consejos.grid(row=3, column=1, sticky="nsew", padx=(0, 10), pady=(5, 10))

    # Simulación de datos (reemplaza con tus datos reales)
    #ejercicios = ["Sentadillas", "Curl Bíceps"]
    tiempo_total = "00:25:30"
    #repeticiones = [random.randint(5, 20) for _ in ejercicios]
    fecha_ultima = "2024-06-10"
    mejor_ejercicio = nombreEjercicio[repeticiones.index(max(repeticiones))]

    # Crear figura de matplotlib
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    barras = ax.bar(nombreEjercicio, repeticiones, color=["#00ADB5", "#393E46"])
    ax.set_ylabel("Repeticiones")
    ax.set_title("Repeticiones por ejercicio")
    ax.set_ylim(0, max(repeticiones) + 5)
    for bar in barras:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), ha='center', va='bottom', fontsize=10)
    fig.tight_layout()

    # Integrar la gráfica en Tkinter
    canvas = FigureCanvasTkAgg(fig, master=frame_grafica)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    # Detalles de la sesión (derecha arriba)
    lbl_detalles = ctk.CTkLabel(frame_detalles, text="Detalles de la sesión", font=("Arial", 16, "bold"), fg_color="#e6eaf8", text_color="#393E46")
    lbl_detalles.pack(pady=(10, 5))
    lbl_fecha = ctk.CTkLabel(frame_detalles, text=f"Última sesión: {fecha_ultima}", font=("Arial", 13), fg_color="#e6eaf8", text_color="#393E46")
    lbl_fecha.pack(pady=5)
    lbl_tiempo = ctk.CTkLabel(frame_detalles, text=f"Tiempo total: {tiempo_total}", font=("Arial", 13), fg_color="#e6eaf8", text_color="#393E46")
    lbl_tiempo.pack(pady=5)
    lbl_mejor = ctk.CTkLabel(frame_detalles, text=f"Ejercicio destacado: {mejor_ejercicio}", font=("Arial", 13), fg_color="#e6eaf8", text_color="#00ADB5")
    lbl_mejor.pack(pady=5)

    # Resumen (derecha medio)
    lbl_resumen = ctk.CTkLabel(frame_resumen, text="Resumen", font=("Arial", 15, "bold"), fg_color="#e6eaf8", text_color="#393E46")
    lbl_resumen.pack(pady=(10, 5))
    lbl_total = ctk.CTkLabel(frame_resumen, text=f"Total de repeticiones: {sum(repeticiones)}", font=("Arial", 13), fg_color="#e6eaf8", text_color="#393E46")
    lbl_total.pack(pady=5)
    lbl_ejercicios = ctk.CTkLabel(frame_resumen, text=f"Ejercicios realizados: {len(nombreEjercicio)}", font=("Arial", 13), fg_color="#e6eaf8", text_color="#393E46")
    lbl_ejercicios.pack(pady=5)

    # Progreso semanal (derecha abajo)
    dias = [(datetime.now() - timedelta(days=i)).strftime("%d/%m") for i in range(6, -1, -1)]
    repeticiones_semanal = [random.randint(5, 20) for _ in dias]

    lbl_progreso = ctk.CTkLabel(frame_progreso, text="Progreso semanal", font=("Arial", 14, "bold"), fg_color="#e6eaf8", text_color="#393E46")
    lbl_progreso.pack(pady=(10, 5))

    fig2, ax2 = plt.subplots(figsize=(3, 1.5), dpi=100)
    ax2.plot(dias, repeticiones_semanal, marker='o', color="#00ADB5")
    ax2.set_ylabel("Reps")
    ax2.set_title("Repeticiones últimos 7 días")
    ax2.set_ylim(0, max(repeticiones_semanal) + 5)
    fig2.tight_layout()
    canvas2 = FigureCanvasTkAgg(fig2, master=frame_progreso)
    canvas2.draw()
    canvas2.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

    # Historial de sesiones (izquierda medio)
    lbl_historial = ctk.CTkLabel(frame_historial, text="Historial de sesiones", font=("Arial", 14, "bold"), fg_color="#e6eaf8", text_color="#393E46")
    lbl_historial.pack(pady=(10, 5))

    tree = ttk.Treeview(frame_historial, columns=("Fecha", "Ejercicio", "Reps"), show="headings", height=6)
    tree.heading("Fecha", text="Fecha")
    tree.heading("Ejercicio", text="Ejercicio")
    tree.heading("Reps", text="Reps")
    for fila in historial:
        tree.insert("", "end", values=fila)
    tree.pack(fill="both", expand=True, padx=5, pady=5)

    # Ranking de ejercicios (izquierda abajo)
    lbl_ranking = ctk.CTkLabel(frame_ranking, text="Ranking de ejercicios", font=("Arial", 14, "bold"), fg_color="#e6eaf8", text_color="#393E46")
    lbl_ranking.pack(pady=(10, 5))

    ranking = [("Curl Bíceps", 32), ("Sentadillas", 28)]
    for i, (ej, reps) in enumerate(ranking, 1):
        lbl = ctk.CTkLabel(frame_ranking, text=f"{i}. {ej} - {reps} reps", font=("Arial", 12), fg_color="#e6eaf8", text_color="#00ADB5" if i == 1 else "#393E46")
        lbl.pack(anchor="w", padx=20)

    # Consejos personalizados (derecha abajo)
    lbl_consejos = ctk.CTkLabel(frame_consejos, text="Consejo personalizado", font=("Arial", 14, "bold"), fg_color="#e6eaf8", text_color="#393E46")
    lbl_consejos.pack(pady=(10, 5))

    consejo = "¡Buen trabajo! Intenta aumentar 2 repeticiones en tu próxima sesión de Sentadillas."
    lbl_consejo_texto = ctk.CTkLabel(frame_consejos, text=consejo, font=("Arial", 12), fg_color="#e6eaf8", text_color="#393E46", wraplength=220, justify="left")
    lbl_consejo_texto.pack(padx=10, pady=(0, 10))

    # Botón para cerrar (abajo)
    btn_cerrar = ctk.CTkButton(
        ventana_stats,
        text="Cerrar",
        fg_color="#00ADB5",
        text_color="white",
        command=salir_stats,
        width=200,
        height=40,  # <-- Aumenta la altura aquí
        font=("Arial", 16, "bold")
    )
    btn_cerrar.pack(pady=(0, 10))

def main():
    global nombreUsuario, password, usuarioID
    threading.Thread(target=cargar_modelo_y_labels).start()
    nombreUsuario, password,usuarioID = cargar_credenciales()
    if nombreUsuario and password:
        conexion = conectar_bd()
        if conexion:
            interfazInicial()
            return
    
    interfazInicial()


if __name__ == "__main__":
    main()