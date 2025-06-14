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
    
    """
def main():
    conexion = conectar_bd()
    if conexion:
        print("✅ Conexión exitosa a la base de datos.")
        cursor = conexion.cursor()
        
        # Aquí puedes realizar consultas o inserciones
        cursor.execute("SELECT * FROM articulos")
        for row in cursor.fetchall():
            print(row)
        
        # Cerrar conexión
        cursor.close()
        conexion.close()
    else:
        print("❌ No se pudo establecer la conexión a la base de datos.")
        """