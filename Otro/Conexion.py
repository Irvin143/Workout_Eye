import pyodbc

def conectar_bd(nombreUsuario,contraseña):
    try:
        conexion = pyodbc.connect(
            f'DRIVER={{ODBC Driver 17 for SQL Server}};'
            f'SERVER=localhost;'
            f'DATABASE=ventas;'
            f'UID={nombreUsuario};'
            f'PWD={contraseña};'
        )
        return conexion
    except Exception as e:
        print("❌ Error al conectar a la base de datos:", e)
        return None
    

def insertar_usuario(conexion, nombre, genero, edad):
    try:
        cursor = conexion.cursor()
        cursor.execute("""
            INSERT INTO Usuarios (Nombre, Genero, Edad)
            VALUES (?, ?, ?)
        """, (nombre, genero, edad))
        conexion.commit()
        print("✅ Usuario insertado correctamente")
        return cursor.lastrowid  # ID del nuevo usuario
    except Exception as e:
        print("❌ Error al insertar usuario:", e)
        return None
    
def consultar_estadisticas(conexion, usuarioID):
    try:
        cursor = conexion.cursor()
        cursor.execute("""
            SELECT Repeticiones, ErroresPostura, PuntajeTecnica
            FROM EstadisticasEjercicio
            WHERE UsuarioID = ?
        """, (usuarioID,))
        fila = cursor.fetchone()
        if fila:
            return {
                "Repeticiones": fila[0],
                "ErroresPostura": fila[1],
                "PuntajeTecnica": float(fila[2])
            }
        else:
            return None
    except Exception as e:
        print("❌ Error al consultar estadísticas:", e)
        return None

def aumentar_repeticiones(conexion, usuarioID, incremento, ErroresPostura, PuntajeTecnica):
    try:
        cursor = conexion.cursor()
        cursor.execute("""
            UPDATE EstadisticasEjercicio
            SET Repeticiones = Repeticiones + ?,
                ErroresPostura = ?,
                PuntajeTecnica = ?
            WHERE UsuarioID = ?
        """, (incremento, ErroresPostura, PuntajeTecnica, usuarioID))
        conexion.commit()
        print("✅ Repeticiones actualizadas correctamente")
        return True
    except Exception as e:
        print("❌ Error al actualizar repeticiones:", e)
        return False