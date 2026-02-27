import threading
from pygrabber.dshow_graph import FilterGraph
import cv2
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
from Otro.Utilidades.Ejercicios.UtilEjercicio import evaluarEjercicio,predecir_ejercicio
from Otro.Utilidades.Ejercicios.EvaluarEjericios import evaluar_sentadilla, evaluar_curl_biceps, evaluar_pullup
from Otro.Utilidades.Utilidades import convertir_landmarks_a_diccionario, detener_animacion, animar_texto
from Otro.Conexion.Conexion import actualizar_estadisticas
import customtkinter as ctk
import requests

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

def mostrar_camara( btn_camara,conexion, usuarioID,opcion,model, le):
    global nombre_ejercicio, zonaError, repeticiones, keypoints_cuerpo,estadisticas,noFrame
    noFrame = 0
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
    frames_list = []
    
    def actualizar_frame():
        global nombre_ejercicio, zonaError, repeticiones, keypoints_cuerpo, estadisticas, noFrame
        ventana_tamaño = 100
        ret, frame = cap.read()
        if not ret:
            ventana_camara.after(10, actualizar_frame)
            return

        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            noFrame += 1
            altura, ancho, _ = frame.shape
            puntos = []
            landmarks = results.pose_landmarks.landmark
            
            frames_list.append(frame)  # Guardar frame en la lista
            
            for lm in results.pose_landmarks.landmark:
                puntos.extend([lm.x, lm.y, lm.z])

            keyCuerpo = convertir_landmarks_a_diccionario(results)
            keypoints_cuerpo.append(keyCuerpo)
            keypoints.append(puntos)
            print("Frame numero;", noFrame)
            evaluarEjercicio(nombre_ejercicio, keypoints_cuerpo, landmarks, ancho, altura, zonaError, frame_rgb)

            if len(keypoints) == 20:
                files = []
                for i, frame_img in enumerate(frames_list):
                    _, buffer = cv2.imencode('.jpg', frame_img)
                    files.append(('frames', (f'frame_{i}.jpg', buffer.tobytes(), 'image/jpeg')))
                
                requests.post("http://localhost:8000/keypoints", files=files, data={"ejercicio": nombre_ejercicio})
                response = requests.post("http://localhost:8000/keypoints", files=files)

                data = response.json()  # ← aquí ya tienes el diccionario

                print(data)
                files.clear()  # Limpiar la lista de archivos después de enviar
                frames_list.clear()  # Limpiar la lista de frames después de enviar
                """
                frames_list.clear()
                zonaError = []
                repeticionesAux = 0
                nombre_ejercicio = predecir_ejercicio(keypoints, model, le)
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
                print(f"Ejercicio detectado: {nombre_ejercicio}")
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
                """
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        lbl_video.imgtk = imgtk
        lbl_video.configure(image=imgtk)
        ventana_camara.after(10, actualizar_frame)

    actualizar_frame()


def cargarInterfazCamaras(btn,ventanaInicial,frameCamaras, opcion):
    global camaras_disponibles
    camaras_disponibles = []

    def cargar_camaras():
        global camaras_disponibles,animar
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

    animar_texto(btn,ventanaInicial)
    threading.Thread(target=cargar_camaras).start()


def cargarCamara(btn,ventanaInicial,btn_camara,conexion, usuarioID,opcion,lbl_txtSeleccion,model, le):
    if opcion.get() == "":
        lbl_txtSeleccion.configure(text_color = "#ff0000")
    else:
        lbl_txtSeleccion.configure(text_color = "#393E46")
        animar_texto(btn,ventanaInicial)
        threading.Thread(target=mostrar_camara, args=(btn_camara, conexion, usuarioID, opcion, model, le)).start()
