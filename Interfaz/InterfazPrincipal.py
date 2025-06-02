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

def main():
    interfaz()

if __name__ == "__main__":
    main()
