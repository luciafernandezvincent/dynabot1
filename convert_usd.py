
import os
from pxr import Usd

def convert_usd():
    filename = "dyna1_gazebo3.usd"
    found_path = None
    
    # 1. Buscar el archivo de forma automática desde la ruta actual hacia abajo
    print(f"Buscando '{filename}' en tu proyecto...")
    for root, dirs, files in os.walk("."):
        if filename in files:
            found_path = os.path.join(root, filename)
            break
            
    if not found_path:
        print(f"Error: No se encontró el archivo '{filename}' en ningún directorio desde aquí.")
        print(f"Directorio de búsqueda actual: {os.getcwd()}")
        return
        
    print(f"¡Archivo encontrado en!: {found_path}")
    print(f" Ruta absoluta: {os.path.abspath(found_path)}")
    
    # 2. Intentar abrir el stage de manera segura
    try:
        stage = Usd.Stage.Open(found_path)
    except Exception as e:
        print(f" Error crítico al intentar abrir el stage: {e}")
        return
        
    if not stage:
        print("Error: El stage de USD no se pudo inicializar (retornó NULL). El archivo podría estar corrupto.")
        return
        
    # 3. Obtener el root layer de manera segura
    root_layer = stage.GetRootLayer()
    if not root_layer:
        print("Error: El Stage no contiene un Root Layer válido.")
        return
        
    # 4. Exportar a formato de texto (.usda)
    output_filename = "dyna1_gazebo3.usda"
    try:
        root_layer.Export(output_filename)
        print(f"\n¡Éxito total! El archivo se convirtió correctamente.")
        print(f" Guardado en: {os.path.abspath(output_filename)}")
    except Exception as e:
        print(f"Error al exportar el layer: {e}")

if __name__ == "__main__":
    convert_usd()
