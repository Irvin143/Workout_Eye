import numpy as np

def veredictoCurl_biceps(keypoints_cuerpo, landmarks, ancho, altura,zonaError):
    marcas_error_global = []
    if len(keypoints_cuerpo) == 0:
        print("No se detectaron poses.")
        return "Desconocido"
    
    if "left_elbow" in zonaError:
        hombro_dx = int(landmarks[15].x * ancho)
        hombro_dy = int(landmarks[15].y * altura)
        codo_dx = int(landmarks[13].x * ancho)
        codo_dy = int(landmarks[13].y * altura)
        marcas_error_global.append(("linea", codo_dx, codo_dy, hombro_dx, hombro_dy))
    if "right_elbow" in zonaError:
        hombro_dx = int(landmarks[16].x * ancho)
        hombro_dy = int(landmarks[16].y * altura)
        codo_dx = int(landmarks[14].x * ancho)
        codo_dy = int(landmarks[14].y * altura)
        marcas_error_global.append(("linea", codo_dx, codo_dy, hombro_dx, hombro_dy))
    
    return marcas_error_global

def evaluar_curl_biceps(keypoints_cuerpo):
    zonaError = []
    angulo_codo_derecho = 0
    angulo_codo_izquierdo = 0

    for i, frame_kp in enumerate(keypoints_cuerpo):
        angulo_codo_derecho = calcular_angulo(
            frame_kp["right_shoulder"],
            frame_kp["right_elbow"],
            frame_kp["right_wrist"]
        )
        angulo_codo_izquierdo = calcular_angulo(
            frame_kp["left_shoulder"],
            frame_kp["left_elbow"],
            frame_kp["left_wrist"]
        )
        # Evaluación básica según el ángulo del codo
        if angulo_codo_derecho < 30 or angulo_codo_izquierdo < 30:
            veredicto = f"Frame {i}: ⚠️ Ángulo ({int(angulo_codo_derecho)}°) demasiado flexionado. Riesgo de usar impulso."
            zonaError.append("right_elbow")
        elif 30 <= angulo_codo_derecho <= 60 or 30 <= angulo_codo_izquierdo <= 60:
            veredicto = f"Frame {i}: ✅ Buena contracción del bíceps ({int(angulo_codo_derecho)}°)."
        elif 150 <= angulo_codo_derecho <= 170 or 150 <= angulo_codo_izquierdo <= 170:
            veredicto = f"Frame {i}: ✅ Buena extensión al bajar ({int(angulo_codo_derecho)}°)."
        elif angulo_codo_derecho > 170 or angulo_codo_izquierdo > 170:
            veredicto = f"Frame {i}: ⚠️ Hiperextensión del codo ({int(angulo_codo_derecho)}°)."
            zonaError.append("left_elbow")
        else:
            veredicto = f"Frame {i}: ℹ️ Ángulo fuera del rango esperado ({int(angulo_codo_derecho)}°)."

    print(veredicto)
    return zonaError

def veredicto_squat(keypoints_cuerpo, landmarks, ancho, altura,zonaError):
    marcas_error_global = []
    if len(keypoints_cuerpo) == 0:
        print("No se detectaron poses.")
        return "Desconocido"
    
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

def evaluar_sentadilla(keypoints_cuerpo):
    angulosDerecho = []
    angulosIzquierdo = []

    angulosDerechoTorso = []
    angulosIzquierdoTorso = []

    frame_min = None
    min_cadera_y = 0  # Inicializar con un valor alto
    aux = 0

    rodillas_ok = True

    for i, frame_kp in enumerate(keypoints_cuerpo):
        anguloDerecho = calcular_angulo(frame_kp["right_hip"],frame_kp["right_knee"], frame_kp["right_ankle"])
        anguloIzquierdo = calcular_angulo(frame_kp["left_hip"], frame_kp["left_knee"],frame_kp["left_ankle"])

        anguloDerechoTorso = calcular_angulo(frame_kp["right_hip"],frame_kp["right_knee"], frame_kp["right_ankle"])
        anguloIzquierdoTorso= calcular_angulo(frame_kp["left_hip"], frame_kp["left_knee"],frame_kp["left_ankle"])
        
        if anguloDerecho < 70 or anguloIzquierdo < 70:
            aux += 1

        if rodillas_ok:
            rodillas_ok = frame_kp["right_knee"][1] < frame_kp["right_ankle"][1] 
        
        angulosIzquierdo.append(anguloIzquierdo)
        angulosDerecho.append(anguloDerecho)

        cadera_y = frame_kp["right_hip"][1]

        if cadera_y > min_cadera_y:  # Y es mayor cuanto más abajo (más profundo)
            min_cadera_y = cadera_y
            frame_min = frame_kp

        angulosDerechoTorso.append(anguloDerechoTorso)
        angulosIzquierdoTorso.append(anguloIzquierdoTorso)

    #profundidad_ok = frame_min["right_ankle"][1] > frame_min["right_knee"][1]
    # Puntos
    cadera = (frame_min["right_hip"][0], frame_min["right_hip"][1])
    rodilla = (frame_min["right_knee"][0], frame_min["right_knee"][1])
    tobillo = (frame_min["right_ankle"][0], frame_min["right_ankle"][1])

    angulo_rodilla = calcular_angulo(cadera, rodilla, tobillo)

    # Umbral común: < 90 grados = buena profundidad
    profundidad_ok = angulo_rodilla < 90

    minimo_anguloDerecho = min(angulosDerecho)
    minimo_anguloIzquierdo = min(angulosIzquierdo)

    # Verificar si el ángulo mínimo es menor a 70 grados

    angulo_promedioDerechoTorso = sum(angulosDerechoTorso) / len(angulosDerechoTorso)
    angulo_promedioIzquierdoTorso = sum(angulosIzquierdoTorso) / len(angulosIzquierdoTorso)

    # espalda vertical
    print("Angulo promedio derecho torso:", angulo_promedioDerechoTorso)
    print("Angulo promedio izquierdo torso:", angulo_promedioIzquierdoTorso)
    print(frame_min["right_ankle"][1], frame_min["right_knee"][1])

    torso_ok = angulo_promedioDerechoTorso < 160 and angulo_promedioIzquierdoTorso < 160

    zonaErorr = []
    if rodillas_ok and profundidad_ok and torso_ok:
        return []
    else:
        errores = []
        if not rodillas_ok:
            errores.append("❌ Ángulo de rodilla incorrecto")
            zonaErorr.append("left_knee")
        if not profundidad_ok:
            errores.append("❌ Baja más la cadera")
            zonaErorr.append("right_hip")
        if not torso_ok:
            errores.append("❌ Mantén la espalda recta")
            zonaErorr.append("back")
        print("Errores encontrados:", errores)
        return zonaErorr

def calcular_angulo(a, b, c):
    """Calcula el ángulo en el punto b entre a y c"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return angle