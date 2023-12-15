# OrdenarFotosMavicPro2AdvanceThermal

Script para ordenar fotos termicas

El script toma las fotos que se dejan en la carpeta "Descarga" ,las copia y las pega en el destino de "Planta", creando una carpeta con el nombre del campo y en su interior cada una de las iteraciones que se realizaron(vuelos).

1. Correr script, Buscar el directorio .../Descargas/"Campo01" (Campo01 hace referencia a la división que tiene la planta)
2. Una vez finalizado el script en el directorio .../"Planta" , estarán las carpetas con los nombres de los campos y en su interior las carpeta con los vuelos realizados.

   Estructura de carpetas
- Imagenes
  - Plantas
    - "Planta"
      - Descargas
        -  Campo01
          - Carpeta01
            - Fotos ... 
          - Carpeta02
            - Fotos ...  
          - Carpeta02
            - Fotos ...
      - "Campo01(Ordenado)"
        - V01
        - V02
        - V03
        - ...
        - ...
     - "Campo02(Ordenado)"
        - V01
        - V02
        - V03
        - ...
        - ...
    - "Campo03(Ordenado)"
        - V01
        - V02
        - V03
        - ...
        - ...
     - "Campo04(Ordenado)"
        - V01
        - V02
        - V03

Librerias utilizadas:

- import shutil
- import os
- import tkinter as tk
- from datetime import datetime, timedelta
- import exifread as exifread
- from PIL import Image
- from tkinter import filedialog
