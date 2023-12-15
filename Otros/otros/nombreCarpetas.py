import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox


def agregar_ceros():
    # Abrir el cuadro de diálogo para seleccionar el directorio
    directorio = filedialog.askdirectory(title="Seleccionar directorio")

    if not directorio:
        messagebox.showwarning("Error", "No se ha seleccionado un directorio.")
        return

    # Obtener la lista de archivos en el directorio
    archivos = os.listdir(directorio)

    # Expresión regular para buscar el número entre paréntesis
    patron = r"\((\d+)\)"

    # Recorrer cada archivo y renombrarlo con ceros a la izquierda en el número entre paréntesis
    for archivo in archivos:
        ruta_original = os.path.join(directorio, archivo)
        if os.path.isfile(ruta_original):
            nombre, extension = os.path.splitext(archivo)
            coincidencias = re.findall(patron, nombre)
            for coincidencia in coincidencias:
                numero = int(coincidencia)
                nuevo_numero = str(numero).zfill(4)  # Agregar ceros a la izquierda para tener 4 dígitos
                nuevo_nombre = nombre.replace(f"({coincidencia})", f"({nuevo_numero})")
                ruta_nuevo_nombre = os.path.join(directorio, nuevo_nombre + extension)
                os.rename(ruta_original, ruta_nuevo_nombre)
                ruta_original = ruta_nuevo_nombre

    messagebox.showinfo("Éxito", "Se han agregado ceros a la izquierda en los números correlativos de los nombres de los archivos.")


# Crear la ventana principal de la aplicación
ventana = tk.Tk()
ventana.title("Agregar ceros a nombres de archivos")
ventana.geometry("300x100")

# Botón para seleccionar el directorio
btn_seleccionar_directorio = tk.Button(ventana, text="Seleccionar directorio", command=agregar_ceros)
btn_seleccionar_directorio.pack(pady=20)

# Iniciar el bucle de eventos de la aplicación
ventana.mainloop()
