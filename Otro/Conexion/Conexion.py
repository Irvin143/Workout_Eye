import pyodbc

def conectar_bd():
    try:
        conexion = pyodbc.connect(
            f'DRIVER={{ODBC Driver 17 for SQL Server}};'
            f'SERVER=localhost;'
            f'DATABASE=WorkoutEyeDB;'
            f'UID=irvin;'
            f'PWD=123;'
        )
        return conexion
    except Exception as e:
        print("❌ Error al conectar a la base de datos:", e)
        return None
    
def consultarUsuario(conexion,usuarioId,password):
    try:
        cursor = conexion.cursor()
        cursor.execute("""
            SELECT usuarioId
            FROM Usuarios
            WHERE UsuarioID = ?
            AND Password = ?
        """, (usuarioId,password))
        fila = cursor.fetchone()
        if fila:
            return fila[0] # Retorna el ID del usuario si existe
        else:
            return None
    except Exception as e:
        print("❌ Error al consultar nombre de usuario:", e)
        return None

def grabarUsuario(conexion, nombre, contrasena, genero = 'M', edad = 18):
    try:
        cursor = conexion.cursor()
        # Si usuarioId es None, pásalo como None para el OUTPUT
        cursor.execute("""
            DECLARE @usuarioID INT;
            EXEC sp_grabar_usuario @UsuarioID = @usuarioID OUTPUT, @Nombre=?, @Contrasena=?, @Genero=?, @Edad=?;
            SELECT @usuarioID;
        """, (nombre, contrasena, genero, edad))
        last_id = cursor.fetchone()[0]
        conexion.commit()
        print("✅ Usuario insertado correctamente con ID:", last_id)
        return last_id
    except Exception as e:
        print("❌ Error al insertar usuario:", e)
        return None
    
def consultar_estadisticas(conexion, usuarioID):
    try:
        cursor = conexion.cursor()
        cursor.execute("""
            SELECT ee.Repeticiones,ee.ErroresPostura,PuntajeTecnica,e.Nombre
            FROM Usuarios u
            INNER JOIN EstadisticasEjercicio ee ON u.UsuarioID = ee.UsuarioID
            inner join Ejercicios e on e.EjercicioID = ee.EjercicioID
            WHERE u.UsuarioID = ?
        """, (usuarioID,))
        filas = cursor.fetchall()
        if filas:
            # Devuelve una lista de diccionarios, uno por cada fila encontrada
            return [
                {
                    "Repeticiones": fila[0],
                    "ErroresPostura": fila[1],
                    "PuntajeTecnica": float(fila[2]),
                    "NombreEjercicio": fila[3]
                }
                for fila in filas
            ]
        else:
            return []
    except Exception as e:
        print("❌ Error al consultar estadísticas:", e)
        return []
    
def actualizar_estadisticas(conexion, usuarioID, nombreEjericcio, repeticiones, erroresPostura, puntajeTecnica):
    try:
        cursor = conexion.cursor()
        cursor.execute("""
            exec sp_actualizar_estadisticas_ejercicio @UsuarioID = ?,@nombre = ?, @repeticiones = ?, @ErroresPostura = ?, @PuntajeTecnica = ?
        """, (usuarioID, nombreEjericcio, repeticiones, erroresPostura, puntajeTecnica))
        conexion.commit()
        print("✅ Estadísticas actualizadas correctamente")
        return True
    except Exception as e:
        print("❌ Error al actualizar estadísticas:", e)
        return False