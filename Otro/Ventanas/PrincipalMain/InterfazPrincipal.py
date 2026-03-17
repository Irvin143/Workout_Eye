import threading
import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from Otro.Conexion.Conexion import conectar_bd
from keras.models import load_model
import pickle
from Otro.Utilidades.Ejercicios.SubirVideos import main as entrenarIA
#from Otro.Utilidades.Ejercicios.Proyecto import main as entrenarIA
from Otro.Ventanas.Secundarias.VentanasSecundarias import interfazInicioSesion, interfaz_mostrar_estadisticas, interfaz_mostrar_notificaciones, interfaz_subir_video
from Otro.Utilidades.Utilidades import centrar_ventana, cargar_credenciales
from Otro.Utilidades.UtilCamaras import cargarCamara, cargarInterfazCamaras

class VentanaPrincipal(ctk.CTk):
    def __init__(self, nombreUsuario, password, usuarioID, model, le, conexion):
        super().__init__(fg_color="#000000")
        self.title("Interfaz Secundaria")
        centrar_ventana(self, 1000, 800)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)
        self.model = model
        self.le = le
        self.conexion = conexion
        self.nombreUsuario = nombreUsuario
        self.password = password
        self.usuarioID = usuarioID

        ruta_script = os.path.dirname(os.path.abspath(__file__))

        # --- Frame Sesión ---
        frameSesion = ctk.CTkFrame(self, corner_radius=20, height=100, fg_color="#c4d2f4")
        frameSesion.columnconfigure(0, weight=1)
        frameSesion.columnconfigure(1, weight=1)
        frameSesion.columnconfigure(2, weight=1)
        frameSesion.grid(row=0, column=0, columnspan=3, sticky="nsew", padx=20, pady=20)
        imagen = Image.open(os.path.join(ruta_script, "..", "..", "..", "Imagenes", "loginWorkout.png"))
        imagen_tk_login = ImageTk.PhotoImage(imagen)
        btn_fotoSesion = ctk.CTkButton(frameSesion, image=imagen_tk_login, text="", width=40, height=40,
                                        fg_color="transparent", hover_color="#a1b9ed",
                                        command=interfazInicioSesion)
        btn_fotoSesion.image = imagen_tk_login
        btn_fotoSesion.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        lbl_datosSesion = ctk.CTkLabel(frameSesion, text=f"Hola, {self.nombreUsuario} ", font=("Arial", 14, "bold"),
                                        fg_color="#c4d2f4", text_color="#393E46")
        lbl_datosSesion.grid(row=0, column=0, padx=(65, 0), pady=10, sticky="w")

        imagen = Image.open(os.path.join(ruta_script, "..", "..", "..", "Imagenes", "campanaWorkout.png"))
        imagen_tk = ImageTk.PhotoImage(imagen)
        btn_notificaciones = ctk.CTkButton(frameSesion, image=imagen_tk, text="", corner_radius=20, width=30,
                                            fg_color="#9eb8f9", command=interfaz_mostrar_notificaciones)
        btn_notificaciones.grid(row=0, column=2, padx=10, pady=10, sticky="e")

        # --- Frame Logo ---
        frameLogo = ctk.CTkFrame(self, corner_radius=20, height=180)
        frameLogo.columnconfigure(0, weight=1)
        frameLogo.columnconfigure(1, weight=1)
        frameLogo.columnconfigure(2, weight=1)
        frameLogo.grid(row=1, column=0, columnspan=3, sticky="nsew", padx=20, pady=20)

        fondo = Image.open(os.path.join(ruta_script, "..", "..", "..", "Imagenes", "fondoWorkout.png")).convert("RGBA")
        logo = Image.open(os.path.join(ruta_script, "..", "..", "..", "Imagenes", "logoWorkout.png")).convert("RGBA")
        x = (fondo.width - logo.width) // 2
        y = (fondo.height - logo.height) // 2
        fondo.paste(logo, (x, y), logo)
        imagen_combinada = ImageTk.PhotoImage(fondo)
        lbl_fondo = ctk.CTkLabel(frameLogo, image=imagen_combinada, text="", height=180)
        lbl_fondo.image = imagen_combinada
        lbl_fondo.place(x=0, y=0, relwidth=1, relheight=1)

        # --- Frame Timer ---
        frameTimer = ctk.CTkFrame(self, corner_radius=20, width=250, height=225, fg_color="#c4d2f4")
        frameTimer.columnconfigure(0, weight=1)
        frameTimer.columnconfigure(1, weight=1)
        frameTimer.columnconfigure(2, weight=1)
        frameTimer.grid_propagate(False)
        frameTimer.grid(row=3, column=0, padx=20, pady=20)

        lbl_timer = tk.Label(frameTimer, text="Tiempo", font=("Arial", 14, "bold"), bg="#c4d2f4", fg="#393E46")
        lbl_timer.grid(row=0, column=0, padx=30, pady=(20, 30), sticky="w")

        imagen = Image.open(os.path.join(ruta_script, "..", "..", "..", "Imagenes", "pngwing.com.png"))
        imagen_tk = ImageTk.PhotoImage(imagen)
        lbl_imagen = tk.Label(frameTimer, image=imagen_tk, bg="#c4d2f4", height=20, width=20)
        lbl_imagen.image = imagen_tk
        lbl_imagen.grid(row=0, column=1, padx=10, pady=(20, 30), sticky="e")

        lbl_fecha = tk.Label(frameTimer, text=self.mostrar_fecha(), font=("Arial", 12, "bold"), bg="#c4d2f4", fg="#393E46")
        lbl_fecha.grid(row=1, column=0, padx=30, pady=20, sticky="w")

        imgaenCalendario = Image.open(os.path.join(ruta_script, "..", "..", "..", "Imagenes", "calendarioWorkout.png"))
        imagen_tkCalendario = ImageTk.PhotoImage(imgaenCalendario)
        lbl_imagenCalendario = tk.Label(frameTimer, image=imagen_tkCalendario, bg="#c4d2f4", height=20, width=20)
        lbl_imagenCalendario.image = imagen_tkCalendario
        lbl_imagenCalendario.grid(row=1, column=1, padx=10, pady=(20, 5), sticky="e")

        self.lbl_hora = tk.Label(frameTimer, text="", font=("Arial", 18, "bold"), bg="#c4d2f4", fg="#393E46")
        self.lbl_hora.grid(row=2, column=0, padx=30, pady=(20, 5), sticky="w")

        lbl_horaTexto = tk.Label(frameTimer, text="Horas", font=("Arial", 12, "bold"), bg="#c4d2f4", fg="#393E46")
        lbl_horaTexto.grid(row=3, column=0, padx=30, sticky="w")

        # --- Frame Estadísticas ---
        frameEstadisticas = ctk.CTkFrame(self, corner_radius=20, width=250, height=225, fg_color="#c4d2f4")
        frameEstadisticas.columnconfigure(0, weight=1)
        frameEstadisticas.grid_propagate(False)
        frameEstadisticas.grid(row=3, column=1, padx=10, pady=20)

        btn_estadisticas = ctk.CTkButton(frameEstadisticas, fg_color="transparent", text="Estadisticas",
                                        font=("Arial", 18, "bold"), corner_radius=20, text_color="black",
                                        command=lambda: interfaz_mostrar_estadisticas(self.usuarioID))
        btn_estadisticas.grid(row=0, column=0, padx=10, pady=(10, 0))

        imagen_estadisticas = Image.open(os.path.join(ruta_script, "..", "..", "..", "Imagenes", "estadisticasWorkout.png"))
        imagen_tk_estadisticas = ImageTk.PhotoImage(imagen_estadisticas)
        lbl_imagen_estadisticas = ctk.CTkLabel(frameEstadisticas, image=imagen_tk_estadisticas, text="")
        lbl_imagen_estadisticas.image = imagen_tk_estadisticas
        lbl_imagen_estadisticas.grid(row=1, column=0, padx=10)

        self.lbl_repeticiones = ctk.CTkLabel(frameEstadisticas, text="Repeticiones: 0 ", font=("Arial", 14, "bold"),
                                            fg_color="#c4d2f4", text_color="#393E46")

        # --- Frame Cámaras ---
        frameCamaras = ctk.CTkFrame(self, corner_radius=20, width=250, height=225, fg_color="#c4d2f4")
        frameCamaras.columnconfigure(0, weight=1)
        frameCamaras.grid_propagate(False)
        frameCamaras.grid(row=3, column=2, padx=20, pady=20)

        lbl_txtCamaras = ctk.CTkLabel(frameCamaras, text="Cámaras disponibles: ", font=("Arial", 14, "bold"),
                                    fg_color="#c4d2f4", text_color="#393E46")
        lbl_txtCamaras.grid(row=0, column=0, padx=10, pady=10)

        self.lbl_txtSeleccion = ctk.CTkLabel(frameCamaras, text="Selecciona una cámara", font=("Arial", 14, "bold"),
                                            fg_color="#c4d2f4", text_color="#393E46")
        self.lbl_txtSeleccion.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="w")

        self.opcion = ctk.StringVar(value="")

        # Botón Iniciar Cámara
        self.btn_camara = ctk.CTkButton(
            self,
            text="Iniciar cámara",
            font=("Arial", 18, "bold"),
            corner_radius=20,
            fg_color="#00ADB5",
            text_color="white",
            width=250,
            height=75,
            command=lambda: cargarCamara(self.btn_camara, self, self.btn_camara, self.conexion, self.usuarioID, self.opcion, self.lbl_txtSeleccion, self.model, self.le)
        )
        self.btn_camara.grid(row=2, column=2, padx=10, pady=10)
        threading.Thread(target=lambda: cargarInterfazCamaras(self.btn_camara, self, frameCamaras, self.opcion)).start()

        # Botón Entrenar Modelo
        btn_entrenar = ctk.CTkButton(
            self,
            text="Entrenar modelo",
            font=("Arial", 18, "bold"),
            corner_radius=20,
            fg_color="#00ADB5",
            text_color="white",
            width=250,
            height=75,
            command=entrenarIA
        )
        btn_entrenar.grid(row=2, column=0, pady=10)

        # Botón Predecir Video
        btn_predecir_video = ctk.CTkButton(
            self,
            text="Predecir video",
            font=("Arial", 18, "bold"),
            corner_radius=20,
            fg_color="#00ADB5",
            text_color="white",
            width=250,
            height=75,
            command=lambda: threading.Thread(target=interfaz_subir_video(self.model, self.le)).start()
        )
        btn_predecir_video.grid(row=2, column=1, pady=10)

        # Botón Cerrar
        btn_cerrar = ctk.CTkButton(
            self,
            text="Cerrar",
            font=("Arial", 18, "bold"),
            corner_radius=20,
            fg_color="#00ADB5",
            text_color="white",
            width=25,
            height=10,
            command=self.destroy
        )
        btn_cerrar.grid(row=4, column=0, padx=10, pady=50, sticky="w")

        self.mostrar_hora()

    def mostrar_fecha(self):
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d")

    def mostrar_hora(self):
        from datetime import datetime
        hora_actual = datetime.now().strftime("%H:%M:%S")
        self.lbl_hora.configure(text=hora_actual)
        self.lbl_hora.after(1000, self.mostrar_hora)

# --- Código para lanzar la ventana principal ---
def cargar_modelo_y_labels():
    global model, le, modelo_listo
    print("Cargando modelo y etiquetas...")
    model = load_model("datos/modelo_ejercicios.h5")
    with open("datos/labels.pkl", "rb") as f:
        le = pickle.load(f)
    print("Modelo y etiquetas cargados.")
    return model, le

def main():
    conexion = conectar_bd()
    model = None
    le = None
    model, le = cargar_modelo_y_labels()
    nombreUsuario, password, usuarioID = cargar_credenciales()
    app = VentanaPrincipal(nombreUsuario, password, usuarioID, model, le, conexion)
    app.mainloop()

if __name__ == "__main__":
    main()