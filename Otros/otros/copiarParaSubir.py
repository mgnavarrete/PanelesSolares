
import os
import shutil


referencias = [
    ["C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/SF1.1PP/para_subir", "C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/paraSubir_dda/219"],
    ["C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/SF1.5PP/para_subir", "C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/paraSubir_dda/220"],
    ["C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/SF2.1PP/para_subir", "C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/paraSubir_dda/221"],
    ["C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/SF2.4PP/para_subir", "C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/paraSubir_dda/222"],
    ["C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/SF1.2PP/para_subir", "C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/paraSubir_dda/244"],
    ["C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/SF1.3PP/para_subir", "C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/paraSubir_dda/245"],
    ["C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/SF1.3_2PP/para_subir", "C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/paraSubir_dda/246"],
    ["C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/SF1.3_3PP/para_subir", "C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/paraSubir_dda/247"],
    ["C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/SF1.4PP/para_subir", "C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/paraSubir_dda/248"],
    ["C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/SF1.6PP/para_subir", "C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/paraSubir_dda/249"],
    ["C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/SF1.7_SF1.10PP/para_subir", "C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/paraSubir_dda/250"],
    ["C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/SF1.8PP/para_subir", "C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/paraSubir_dda/251"],
    ["C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/SF1.9PP/para_subir", "C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/paraSubir_dda/252"],
    ["C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/SF1.11PP/para_subir", "C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/paraSubir_dda/253"],
    ["C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/SF1.12PP/para_subir", "C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/paraSubir_dda/254"],
    ["C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/SF1.13PP/para_subir", "C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/paraSubir_dda/255"],
    ["C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/SF1.14PP/para_subir", "C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/paraSubir_dda/256"],
    ["C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/SF2.2PP/para_subir", "C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/paraSubir_dda/257"],
    ["C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/SF2.3PP/para_subir", "C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/paraSubir_dda/258"],
    ["C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/SF2.5PP/para_subir", "C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/paraSubir_dda/259"],
    ["C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/SF2.6PP/para_subir", "C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/paraSubir_dda/260"],
    ["C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/SF2.7PP/para_subir", "C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/paraSubir_dda/261"],
    ["C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/SF2.8PP/para_subir", "C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/paraSubir_dda/262"],
    ["C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/SF2.9PP/para_subir", "C:/Users/Adentu/Desktop/Adentu Paneles/Imagenes/Diego de almagro/paraSubir_dda/263"],
]

for referencia in referencias:
    directorio_origen = referencia[0]
    directorio_destino = referencia[1]
    archivos = os.listdir(directorio_origen)

    for archivo in archivos:
        ruta_origen = os.path.join(directorio_origen, archivo)
        ruta_destino = os.path.join(directorio_destino, archivo)
        shutil.copy2(ruta_origen, ruta_destino)

