import os
from datetime import datetime
import tkinter as tk
from tkinter import filedialog


import simplekml
import exif_gps
import exifread as exifread
from PIL import Image, ExifTags

# Crear una ventana Tkinter solo para seleccionar el directorio
screen = tk.Tk()
screen.withdraw()  # Ocultar la ventana principal de tkinter
pathMain = filedialog.askdirectory() # Lectura de directorio
listFiles = os.listdir(pathMain)    # Crea un listado de los archivos de la carpeta

# Creación de nombres -> nomenclatura planta -> Campo a renombrar
prefix_pfv = input("Ingrese nomenclatura de planta: ")
prefix_division = input("Ingrese nomenclatura de campo: ")

prefix_name = prefix_pfv+"_"+prefix_division

def extract_datatime(pathFile):
    im = Image.open(pathFile)  # Toma el archivo
    tags = {}  # Crea una lista vacia para almacenar los tags
    with open(pathFile, "rb") as file_handle:  # Abre el archivo
        tags = exifread.process_file(
            file_handle)  # Guarda los metadatos en formato lista 'EXIF DateTimeOriginal':'2023:10:05 13:13:04'
        if "EXIF DateTimeOriginal" in tags.keys():  # Pregunta si hay un tag con este nombre
            datetime_origin = tags["EXIF DateTimeOriginal"]
            input_datetime = datetime.strptime(str(datetime_origin),
                                               "%Y:%m:%d %H:%M:%S") # cambiando el formato de fecha y hora
            output_datetime = input_datetime.strftime("%Y-%m-%d_%H-%M-%S")
            return output_datetime

def filename_prefix_datatime(path,prefix,data,extension,iteration): # DJI_0004_T -> PFV_CT1_0004_T_2023-10-05_13-13-06.JPG
    i = str(iteration).zfill(5)
    path_main = path[:10]   # Extrae el nombre del archivo antes de hasta antes de su extension "DJI_00001_T"
    datatime_temp = data    # Datatime
    new_filename = f'{prefix}{"_"}{i}{path_main[8:]}_{data}{extension}' # Une el nombre del archivo junto con su datatime DJI_0002_T_2023-10-05_13-13-04.JPG
    #print(path+" -> "+new_filename)
    #pathFile = os.path.join(pathMain, data).replace("\\", "/")

def extension_checker(file):    # Devuelve true si la extensión del archivo pertenece a la lista '.jpg','.jpeg','.png','.gif','.bmp','.tiff'
    ext_image = ['.jpg','.jpeg','.png','.gif','.bmp','.tiff']
    name, extension = os.path.splitext(file.lower())
    return extension in ext_image

def extract_coordenates(pathFile):
    im = Image.open(pathFile)  # Toma el archivo
    tags1 = {}  # Crea una lista vacia para almacenar los tags
    with open(pathFile, "rb") as file_handle:  # Abre el archivo
        tags1 = exifread.process_file(file_handle)
        # print(exif_gps.get_exif_location(tags1))
        lat, lon = str(exif_gps.get_exif_location(tags1))[1:-1].split(",")
        return lon,lat
kml = simplekml.Kml()
cont = 1
for subfolders in listFiles:
    path_iterable = os.path.join(pathMain, subfolders).replace("\\", "/")
    if(os.path.isdir(path_iterable)):
        print(path_iterable)
        for file in os.listdir(path_iterable):
            pathFile = os.path.join(path_iterable, file).replace("\\", "/")
            if (extension_checker(pathFile)):
                # fol.newpoint(name=file, coords=[(extract_coordenates(pathFile))])
                datatime = extract_datatime(pathFile)  # Extrae fecha y hora '2023-10-05_13-13-04'
                file_name, extension = os.path.splitext(
                    file)  # separa el nombre del archivo y la extensión "DJI_00001_T",".JPG"
                filename_prefix_datatime(file_name,prefix_name, datatime,
                                  extension,cont)  # Concatena nombre + (horafecha) + extension "DJI_0002_T_2023-10-05_13-13-04.JPG"

                cont=cont+1

                kml.newpoint(name=file, coords=[(extract_coordenates(pathFile))])
kml.save(prefix_name+".kml")
"""
#kml = simplekml.Kml()
#fol = kml.newfolder(name="folderTest")
#fol.open = 1

for file in listFiles:
    pathFile = os.path.join(pathMain, file).replace("\\", "/")
    if(extension_checker(pathFile)):
        #fol.newpoint(name=file, coords=[(extract_coordenates(pathFile))])
        datatime = extract_datatime(pathFile)   # Extrae fecha y hora '2023-10-05_13-13-04'
        file_name,extension = os.path.splitext(file)    # separa el nombre del archivo y la extensión "DJI_00001_T",".JPG"
        filename_datatime(file_name, datatime, extension) # Concatena nombre + (horafecha) + extension "DJI_0002_T_2023-10-05_13-13-04.JPG"


kml.save("test2.kml")

kml = simplekml.Kml()

pnt = kml.newpoint(name='DJI_0001')
pnt.coords=[(long,lat)]
pnt.hit

print(kml.kml("test.kml"))
"""