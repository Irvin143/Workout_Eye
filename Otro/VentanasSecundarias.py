
import threading
from tkinter import Image, filedialog, ttk
from tkinter import messagebox
import tkinter as tk
import customtkinter as ctk
import os
from PIL import Image, ImageTk
from Utilidades import guardar_credenciales, centrar_ventana
from Conexion import consultar_estadisticas, grabarUsuario, consultarUsuario
from Predecir import predecirVideo
from Conexion import conectar_bd

conexion = conectar_bd()

def interfazInicioSesion():
    # Crear ventana de inicio de sesión
    ventana_login = ctk.CTkToplevel()
    ventana_login.title("Inicio de Sesión")
    ventana_login.resizable(False, False)
    ventana_login.grab_set()
    ventana_login.focus_force()
    centrar_ventana(ventana_login, 450, 550)
    ventana_login.configure(fg_color="#c4d2f4")

    # Fondo decorativo
    ruta_script = os.path.dirname(os.path.abspath(__file__))

    # Marco para el formulario
    frame_form = ctk.CTkFrame(ventana_login, corner_radius=20, fg_color="#ffffff", width=350, height=300)
    frame_form.place(relx=0.5, rely=0.5, anchor="center")

    # Logo
    try:
        logo = Image.open(os.path.join(ruta_script, "..", "Imagenes", "logoWorkout.png"))
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



def interfaz_mostrar_estadisticas(usuarioID):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import random
    from datetime import datetime, timedelta

    diccEstadisticas = []
    historial = []
    repeticiones, erroresPostura, puntajeTecnica,nombreEjercicio = [],[],[],[]
    diccEstadisticas = consultar_estadisticas(conexion, usuarioID)
    
    # Obtener los dos ejercicios con más repeticiones
    top_ejercicios = sorted(diccEstadisticas, key=lambda x: int(x['Repeticiones']), reverse=True)[:2]
    ranking = []
    for ejercicio in top_ejercicios:
        nombre = ejercicio['NombreEjercicio']
        reps = int(ejercicio['Repeticiones'])
        ranking.append((nombre, reps))

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
    ventana_stats.grab_set()
    ventana_stats.focus_force()
    ventana_stats.title("Estadísticas de Ejercicios")
    # Maximiza la ventana pero sin quitar la barra de título ni poner fullscreen real
    ventana_stats.state('zoomed')  # Para Windows: maximiza la ventana
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



#Ventana emergente de notificaciones/noticias
def interfaz_mostrar_notificaciones():
    ventana_popup = ctk.CTkToplevel()
    ventana_popup.title("Notificaciones")
    ventana_popup.grab_set()
    ventana_popup.focus_force()
    ventana_popup.resizable(False, False)
    ventana_popup.configure(fg_color="#e6eaf8")
    ventana_popup.overrideredirect(True)  # Elimina la barra de título
    centrar_ventana(ventana_popup, 400, 300)

    lbl_titulo = ctk.CTkLabel(ventana_popup, text="Noticias y Notificaciones", font=("Arial", 16, "bold"), fg_color="#e6eaf8", text_color="#393E46")
    lbl_titulo.pack(pady=(20, 10))

    # Aquí puedes agregar tus noticias/notificaciones
    noticias = [
        "¡Nuevo ejercicio disponible: Press de banca!",
        "Recuerda mantener una buena postura durante tus entrenamientos.",
        "Actualización: Mejoras en la detección de repeticiones.",
        "¡Sigue así! Tu progreso es excelente."
    ]
    for noticia in noticias:
        lbl_noticia = ctk.CTkLabel(ventana_popup, text=f"• {noticia}", font=("Arial", 12), fg_color="#e6eaf8", text_color="#393E46", wraplength=350, justify="left")
        lbl_noticia.pack(anchor="w", padx=20, pady=2)

    btn_cerrar_popup = ctk.CTkButton(ventana_popup, text="Cerrar", fg_color="#00ADB5", text_color="white", command=ventana_popup.destroy)
    btn_cerrar_popup.pack(pady=20)



# Interfaz para subir video y detectar ejercicio
def interfaz_subir_video(model,le,modelo_listo):

    def procesar_video(ruta_video):
        ejercicio = "Desconocido"
        if not modelo_listo:
            messagebox.showwarning("Advertencia", "El modelo no está listo. Por favor, espera a que se cargue el modelo.")
            return
        try:
            ejercicio = predecirVideo(model, le, ruta_video)
            lbl_resultado.configure(text=f"Ejercicio detectado: {ejercicio}", text_color="#00ADB5")
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error al procesar el video:\n{e}")

    ventana_subir = ctk.CTkToplevel()
    ventana_subir.title("Subir Video para Detección")
    ventana_subir.geometry("400x250")
    ventana_subir.grab_set()
    ventana_subir.focus_force()
    ventana_subir.configure(fg_color="#e6eaf8")

    lbl_titulo = ctk.CTkLabel(ventana_subir, text="Subir un video para detectar ejercicio", font=("Arial", 16, "bold"), fg_color="#e6eaf8", text_color="#393E46")
    lbl_titulo.pack(pady=(20, 10))

    ruta_video = ctk.StringVar()

    entry_ruta = ctk.CTkEntry(ventana_subir, textvariable=ruta_video, placeholder_text="Ruta del video...", width=250)
    entry_ruta.pack(pady=10, padx=20)

    def seleccionar_video():
        archivo = filedialog.askopenfilename(
            filetypes=[("Archivos de video", "*.mp4 *.avi *.mov *.mkv"), ("Todos los archivos", "*.*")]
        )
        if archivo:
            ruta_video.set(archivo)

    btn_explorar = ctk.CTkButton(ventana_subir, text="Seleccionar video", fg_color="#00ADB5", text_color="white", command=seleccionar_video)
    btn_explorar.pack(pady=5)

    btn_procesar = ctk.CTkButton(
        ventana_subir,
        text="Procesar video",
        fg_color="#00ADB5",
        text_color="white",
        command=lambda: threading.Thread(target=lambda: procesar_video(ruta_video.get()) if ruta_video.get() else messagebox.showwarning("Advertencia", "Selecciona un video primero.")).start()
    )
    btn_procesar.pack(pady=10)

    lbl_resultado = ctk.CTkLabel(ventana_subir, text="", font=("Arial", 14, "bold"), fg_color="#e6eaf8", text_color="#393E46")
    lbl_resultado.pack(pady=10)

    btn_cerrar = ctk.CTkButton(ventana_subir, text="Cerrar", fg_color="#393E46", text_color="white", command=ventana_subir.destroy)
    btn_cerrar.pack(pady=10)
