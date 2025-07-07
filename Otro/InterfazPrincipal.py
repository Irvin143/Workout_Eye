import threading
import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # '0' = all logs, '1' = filter INFO, '2' = filter WARNING, '3' = only ERROR
from Conexion import conectar_bd
from keras.models import load_model
import pickle
from Proyecto import main as entrenarIA
from VentanasSecundarias import interfazInicioSesion, interfaz_mostrar_estadisticas, interfaz_mostrar_notificaciones, interfaz_subir_video
from Utilidades import  centrar_ventana, cargar_credenciales
from UtilCamaras import cargarCamara, cargarInterfazCamaras

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
    btn_camara.configure(command = lambda: cargarCamara(btn_camara,ventanaInicial,btn_camara,conexion, usuarioID,opcion,lbl_txtSeleccion))
    threading.Thread(target=cargarInterfazCamaras(btn_camara,ventanaInicial,frameCamaras,opcion)).start() 

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