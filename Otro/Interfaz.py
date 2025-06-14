import tkinter as tk
from tkinter import ttk
import customtkinter as ctk
from PIL import Image, ImageTk
import os
from PIL import Image

def interfazInicial():
    global lbl_hora
    # Obtener la ruta absoluta del script
    ruta_script = os.path.dirname(os.path.abspath(__file__))

    # Construir la ruta absoluta de la imagen
    ruta_imagen = os.path.join(ruta_script, "pngwing.com.png")
    ruta_imagenCalendario = os.path.join(ruta_script, "calendarioWorkout.png")

    ventana = ctk.CTk(fg_color="#FFFFFF")  # Fondo oscuro
    ventana.title("Interfaz Secundaria")
    ventana.geometry("1000x800")
    ventana.columnconfigure(0, weight=1)
    ventana.columnconfigure(1, weight=1)
    ventana.columnconfigure(2, weight=1)

    # Botón Iniciar Sesión (arriba izquierda)
    btn_sesion = tk.Button(ventana, text="Iniciar sesión")
    btn_sesion.grid(row=0, column=0, padx=10, pady=10, sticky="w")

    lbl_espacio = tk.Label(ventana, text="",width=25,height=10)
    lbl_espacio.grid(row=1, column=1, padx=10, pady=10)

    frameTimer = ctk.CTkFrame(ventana, corner_radius=20, width=250, height=225,fg_color="#c4d2f4")
    frameTimer.columnconfigure(0, weight=1)
    frameTimer.columnconfigure(1, weight=1)
    frameTimer.columnconfigure(2, weight=1)
    frameTimer.grid_propagate(False)
    frameTimer.grid(row=3, column=0, padx=20, pady=20)

    lbl_timer = tk.Label(frameTimer, text="Tiempo", font=("Arial", 14, "bold"), bg="#c4d2f4", fg="#393E46")
    lbl_timer.grid(row=0, column=0, padx = 30,pady=(20,30),sticky="w")

    imagen = Image.open(ruta_imagen)
    imagen_tk = ImageTk.PhotoImage(imagen)
    lbl_imagen = tk.Label(frameTimer, image=imagen_tk, bg="#c4d2f4",height=20, width=20)
    lbl_imagen.image = imagen_tk  # Mantener una referencia a la imagen
    lbl_imagen.grid(row=0, column=1, padx=10, pady=(20,30), sticky="e")

    lbl_fecha = tk.Label(frameTimer, text= mostrar_fecha(), font=("Arial", 12, "bold"), bg="#c4d2f4", fg="#393E46")
    lbl_fecha.grid(row=1, column=0, padx = 30,pady=20,sticky="w")

    imgaenCalendario = Image.open(ruta_imagenCalendario)
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
    height=75
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
    height=75
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