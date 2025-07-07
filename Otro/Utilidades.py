import json
import os

import numpy as np
import cv2
import mediapipe as mp

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

def centrar_ventana(ventana, ancho, alto):
    ventana.update_idletasks()  # Asegura que obtenga tamaño de pantalla real
    screen_width = ventana.winfo_screenwidth()
    screen_height = ventana.winfo_screenheight()

    x = (screen_width // 2) - (ancho // 2)
    y = (screen_height // 2) - (alto // 2)

    ventana.geometry(f"{ancho}x{alto}+{x}+{y}")
