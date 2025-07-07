import json
import os

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


def animar_texto(btn,ventana):
    # Usa un atributo en el botón para controlar la animación individual
    btn.animar = True
    textoAnterior = btn.cget("text")
    def ciclo(i=0):
        if getattr(btn, "animar", False):
            btn.configure(state="disabled")
            puntos = "." * (i % 4)
            btn.configure(text=f"Cargando{puntos}")
            ventana.after(300, ciclo, i + 1)    
        else:
            btn.configure(text=textoAnterior)
            btn.configure(state="normal")
    ciclo()

def detener_animacion(btn):
    btn.animar = False
    

def convertir_landmarks_a_diccionario(results):
    mediapipe_keypoints = {
        0: "nose",
        1: "left_eye_inner",
        2: "left_eye",
        3: "left_eye_outer",
        4: "right_eye_inner",
        5: "right_eye",
        6: "right_eye_outer",
        7: "left_ear",
        8: "right_ear",
        9: "mouth_left",
        10: "mouth_right",
        11: "left_shoulder",
        12: "right_shoulder",
        13: "left_elbow",
        14: "right_elbow",
        15: "left_wrist",
        16: "right_wrist",
        17: "left_pinky",
        18: "right_pinky",
        19: "left_index",
        20: "right_index",
        21: "left_thumb",
        22: "right_thumb",
        23: "left_hip",
        24: "right_hip",
        25: "left_knee",
        26: "right_knee",
        27: "left_ankle",
        28: "right_ankle"
    }
    puntos_xy = {}
    if results.pose_landmarks:
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            nombre = mediapipe_keypoints.get(idx, f"punto_{idx}")
            puntos_xy[nombre] = [landmark.x, landmark.y]
    return puntos_xy
