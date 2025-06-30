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
    repeticiones = 0
    estado = ""
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
        print(angulo_codo_derecho)
        if angulo_codo_derecho < 55:  # brazo flexionado
            estado = "arriba"
        elif angulo_codo_derecho > 120 and estado == "arriba":
            estado = "abajo"
            repeticiones += 1
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
    return zonaError,repeticiones

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
    repeticiones = 0
    angulosDerecho = []
    angulosIzquierdo = []

    angulosDerechoTorso = []
    angulosIzquierdoTorso = []

    frame_min = None
    min_cadera_y = 0  # Inicializar con un valor alto

    rodillas_ok = True

    estado = "arriba"  # Estado inicial: de pie

    for i, frame_kp in enumerate(keypoints_cuerpo):
        anguloDerecho = calcular_angulo(frame_kp["right_hip"], frame_kp["right_knee"], frame_kp["right_ankle"])
        anguloIzquierdo = calcular_angulo(frame_kp["left_hip"], frame_kp["left_knee"], frame_kp["left_ankle"])

        anguloDerechoTorso = calcular_angulo(frame_kp["right_shoulder"], frame_kp["right_hip"], frame_kp["right_knee"])
        anguloIzquierdoTorso = calcular_angulo(frame_kp["left_shoulder"], frame_kp["left_hip"], frame_kp["left_knee"])

        angulosIzquierdo.append(anguloIzquierdo)
        angulosDerecho.append(anguloDerecho)

        angulosDerechoTorso.append(anguloDerechoTorso)
        angulosIzquierdoTorso.append(anguloIzquierdoTorso)

        cadera_y = frame_kp["right_hip"][1]

        if cadera_y > min_cadera_y:
            min_cadera_y = cadera_y
            frame_min = frame_kp

        # Contar repeticiones: bajada y subida
        # Bajada: ángulo de rodilla < 90 (sentadilla abajo)
        # Subida: ángulo de rodilla > 150 (de pie)
        if estado == "arriba" and (anguloDerecho < 110 or anguloIzquierdo < 100):
            estado = "abajo"
        elif estado == "abajo" and (anguloDerecho > 140 and anguloIzquierdo > 140):
            estado = "arriba"
            repeticiones += 1

        if rodillas_ok:
            rodillas_ok = frame_kp["right_knee"][1] < frame_kp["right_ankle"][1]

    # Puntos
    if frame_min is not None:
        cadera = (frame_min["right_hip"][0], frame_min["right_hip"][1])
        rodilla = (frame_min["right_knee"][0], frame_min["right_knee"][1])
        tobillo = (frame_min["right_ankle"][0], frame_min["right_ankle"][1])

        angulo_rodilla = calcular_angulo(cadera, rodilla, tobillo)
        profundidad_ok = angulo_rodilla < 90
    else:
        profundidad_ok = False

    angulo_promedioDerechoTorso = sum(angulosDerechoTorso) / len(angulosDerechoTorso) if angulosDerechoTorso else 0
    angulo_promedioIzquierdoTorso = sum(angulosIzquierdoTorso) / len(angulosIzquierdoTorso) if angulosIzquierdoTorso else 0

    torso_ok = angulo_promedioDerechoTorso < 160 and angulo_promedioIzquierdoTorso < 160

    zonaErorr = []
    if rodillas_ok and profundidad_ok and torso_ok:
        return zonaErorr, repeticiones
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

        return zonaErorr, repeticiones
    
def veredicto_pullup(keypoints_cuerpo, landmarks, ancho, altura,zonaError):
    marcas_error_global = []
    if len(keypoints_cuerpo) == 0:
        print("No se detectaron poses.")
        return "Desconocido"
    
    if "left_wrist" in zonaError:
        muñeca_dx = int(landmarks[15].x * ancho)
        muñeca_dy = int(landmarks[15].y * altura)
        marcas_error_global.append(("punto", muñeca_dx, muñeca_dy))
    if "right_wrist" in zonaError:
        muñeca_dx = int(landmarks[16].x * ancho)
        muñeca_dy = int(landmarks[16].y * altura)
        marcas_error_global.append(("punto", muñeca_dx, muñeca_dy))
    if "left_elbow" in zonaError:
        codo_dx = int(landmarks[13].x * ancho)
        codo_dy = int(landmarks[13].y * altura)
        marcas_error_global.append(("punto", codo_dx, codo_dy))
    if "right_elbow" in zonaError:
        codo_dx = int(landmarks[14].x * ancho)
        codo_dy = int(landmarks[14].y * altura)
        marcas_error_global.append(("punto", codo_dx, codo_dy))
    
    return marcas_error_global

def evaluar_pullup(keypoints_cuerpo):
    """
    Evalúa si una dominada (pull-up) está bien hecha.
    Retorna zonaError (lista de zonas con error) y repeticiones (int).
    """
    zonaError = []
    repeticiones = 0
    estado = "abajo"  # Estado inicial: colgado
    veredicto = ""
    for i, frame_kp in enumerate(keypoints_cuerpo):
        # Ángulo del codo derecho e izquierdo
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
        # Altura de la barbilla respecto a los hombros
        barbilla_y = frame_kp.get("nose", (0, 0))[1]
        hombro_izq_y = frame_kp["left_shoulder"][1]
        hombro_der_y = frame_kp["right_shoulder"][1]
        hombros_y = (hombro_izq_y + hombro_der_y) / 2

        # Detectar subida (barbilla arriba de los hombros)
        if estado == "abajo" and barbilla_y < hombros_y:
            estado = "arriba"
        # Detectar bajada (barbilla baja de los hombros y codos extendidos)
        elif estado == "arriba" and barbilla_y > hombros_y and angulo_codo_derecho > 150 and angulo_codo_izquierdo > 150:
            estado = "abajo"
            repeticiones += 1

        # Evaluación básica de errores
        if angulo_codo_derecho < 30 or angulo_codo_izquierdo < 30:
            veredicto = f"Frame {i}: ⚠️ Codos demasiado flexionados, posible impulso."
            zonaError.append("right_elbow")
            zonaError.append("left_elbow")
        elif angulo_codo_derecho > 170 or angulo_codo_izquierdo > 170:
            veredicto = f"Frame {i}: ⚠️ Hiperextensión de codo."
            zonaError.append("right_elbow")
            zonaError.append("left_elbow")
        else:
            veredicto = f"Frame {i}: ✅ Movimiento correcto."
    print(veredicto)
    return zonaError, repeticiones

def calcular_angulo(a, b, c):
    """Calcula el ángulo en el punto b entre a y c"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return angle