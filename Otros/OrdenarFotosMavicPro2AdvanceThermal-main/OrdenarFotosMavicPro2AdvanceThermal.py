import shutil
import os
import tkinter as tk
from datetime import datetime, timedelta

import exifread as exifread
from PIL import Image
from tkinter import filedialog


# Nomenclatura
# CODPLANTA_ZONA_CAMPO_VUELO_NOMBREIMAGEN_CORRELATIVOVUELO
# CODPLANTA_ZONA_CAMPO_VUELO_NOMBREIMAGEN_TIEMPO
# Ejemplo: DMK_Z2_PV03_02_DJI_405_T_202310091534

def insertar_texto(cadena, texto):
    izquierda = cadena[:-4]
    derecha = cadena[-4:]
    return '{}{}{}'.format(izquierda, texto, derecha)


def extraer_datatime(path):
    im = Image.open(path)
    tags = {}
    with open(path, "rb") as file_handle:
        tags = exifread.process_file(file_handle)
        if "EXIF DateTimeOriginal" in tags.keys():
            datetime_origin = tags["EXIF DateTimeOriginal"]
            input_datatime = datetime.strptime(str(datetime_origin), "%Y:%m:%d %H:%M:%S")
            output_datatime = input_datatime.strftime("%Y%m%d%H%M%S")
            return output_datatime

def verificador_extension(file):
    ext_image = ['.jpg','.jpeg','.png','.gif','.bmp','.tiff']
    name, ext = os.path.splitext(file.lower())
    return ext in ext_image

def copiar_imagenes(dir0, str_Planta, str_CT, str_V,
                    im_path,it):  # dir0=directorioCampoVuelo , str_planta= ,str_CT= ,str_V= ,im_path=# D:/ProcesamientoEnel_2023-10/Imagenes/Plantas/VDS/NCU20/105MEDIA
    carpeta = it
    vuelo = f'{"V"+str(carpeta).zfill(2)}'  # Numero de carpeta -> V01
    #Tdir = f'{dir0}/{str_Planta}/T/{str_CT}/{vuelo}'  # Path completo
    #Tdir = f'{dir0}/{str_Planta}/T/{str_CT}/{vuelo}' # D/ProcesamientoEnel_2023-10/Imagenes/Plantas/VDS/T/NCU20/100MEDIA
    #Tdir = f'{dir0}/{str_Planta}/T/{str_CT}'
    #RGBdir = f'{dir0}/{str_Planta}/RGB/{str_CT}'#/{vuelo}'
    #os.makedirs(Tdir, exist_ok=True)  # Verifica si existe el path
    #os.makedirs(RGBdir, exist_ok=True)

    #cantfot = 0

    fechaHora_anterior = None
    temp_archivos = sorted(os.listdir(im_path))
    for Pic in temp_archivos:
        if (verificador_extension(Pic)):
            Picdir = im_path + '/' + Pic  # Ruta original de foto -> D:/ProcesamientoEnel_2023-10/Imagenes/Plantas/VDS/NCU20/105MEDIA/DJI_0001_T.JPG
            nombreImagen = Pic.split('.')[0] # Guarda el nombre del archivo sin extensión
            fechaHora = extraer_datatime(Picdir)    # formato de fecha y hora -> 2023-10-04 12:13:39

            if (fechaHora_anterior is not None):
                datatime_actual = datetime.strptime(fechaHora,"%Y%m%d%H%M%S")
                datatime_anterior = datetime.strptime(fechaHora_anterior,"%Y%m%d%H%M%S")
                diferencia_datatime = datatime_actual-datatime_anterior
                timedelta_temp = timedelta(seconds=25)
                if(diferencia_datatime > timedelta_temp):
                    carpeta += 1
                    print("Numero de vuelo: "+str(carpeta).zfill(2))
                    # print("\t- "+Picdir+ " - " + str(datetime.strptime(fechaHora,"%Y%m%d%H%M%S")))
                    if Pic[-6:] == "_T.JPG":
                        nombreFinal = f'{str_Planta}_{str_CT}_{"V" + str(carpeta).zfill(2)}_{nombreImagen}_{fechaHora}.JPG'  # VDS_NCU20_V01_DJI_0212_T_202310091534.JPG
                        temp = f'{dir0}/{str_CT}/{"V" + str(carpeta).zfill(2)}'
                        os.makedirs(temp, exist_ok=True)
                        shutil.copy(Picdir, temp)   # Hace copia de 'D:/ProcesamientoEnel_2023-10/Imagenes/Plantas/VDS/NCU20/100MEDIA/DJI_0651_T.JPG' ->
                        os.rename(f'{temp}/{Pic}', f'{temp}/{nombreFinal}')
                    elif str("." + Pic.split(".")[1]) == ".JPG":
                        nombreFinal = f'{str_Planta}_{str_CT}_{"V" + str(carpeta).zfill(2)}_{nombreImagen + "_T"}_{fechaHora}.JPG'
                        # temp = f'{dir0}/{str_Planta}/T/{str_CT}/{"V" + str(carpeta).zfill(2)}'
                        temp = f'{dir0}/{str_CT}/{"V" + str(carpeta).zfill(2)}'
                        os.makedirs(temp, exist_ok=True)
                        shutil.copy(Picdir,
                                    temp)  # Hace copia de 'D:/ProcesamientoEnel_2023-10/Imagenes/Plantas/VDS/NCU20/100MEDIA/DJI_0651_T.JPG' ->
                        os.rename(f'{temp}/{Pic}', f'{temp}/{nombreFinal}')
                    fechaHora_anterior = fechaHora
                else:
                    if Pic[-6:] == "_T.JPG":
                        nombreFinal = f'{str_Planta}_{str_CT}_{"V" + str(carpeta).zfill(2)}_{nombreImagen}_{fechaHora}.JPG'  # VDS_NCU20_V01_DJI_0212_T_202310091534.JPG
                        temp = f'{dir0}/{str_CT}/{"V" + str(carpeta).zfill(2)}'
                        os.makedirs(temp, exist_ok=True)
                        shutil.copy(Picdir, temp)   # Hace copia de 'D:/ProcesamientoEnel_2023-10/Imagenes/Plantas/VDS/NCU20/100MEDIA/DJI_0651_T.JPG' ->
                        os.rename(f'{temp}/{Pic}', f'{temp}/{nombreFinal}')
                    elif str("." + Pic.split(".")[1]) == ".JPG":
                        nombreFinal = f'{str_Planta}_{str_CT}_{"V" + str(carpeta).zfill(2)}_{nombreImagen + "_T"}_{fechaHora}.JPG'
                        # temp = f'{dir0}/{str_Planta}/T/{str_CT}/{"V" + str(carpeta).zfill(2)}'
                        temp = f'{dir0}/{str_CT}/{"V" + str(carpeta).zfill(2)}'
                        os.makedirs(temp, exist_ok=True)
                        shutil.copy(Picdir,
                                    temp)  # Hace copia de 'D:/ProcesamientoEnel_2023-10/Imagenes/Plantas/VDS/NCU20/100MEDIA/DJI_0651_T.JPG' ->
                        os.rename(f'{temp}/{Pic}', f'{temp}/{nombreFinal}')
                    fechaHora_anterior = fechaHora
            else:
                print("Numero de vuelo: " + str(carpeta).zfill(2))

                if Pic[-6:] == "_T.JPG":  # Path completo
                    nombreFinal = f'{str_Planta}_{str_CT}_{"V" + str(carpeta).zfill(2)}_{nombreImagen}_{fechaHora}.JPG'
                    #temp = f'{dir0}/{str_Planta}/T/{str_CT}/{"V" + str(carpeta).zfill(2)}'
                    temp = f'{dir0}/{str_CT}/{"V" + str(carpeta).zfill(2)}'
                    os.makedirs(temp, exist_ok=True)
                    shutil.copy(Picdir,
                                temp)  # Hace copia de 'D:/ProcesamientoEnel_2023-10/Imagenes/Plantas/VDS/NCU20/100MEDIA/DJI_0651_T.JPG' ->
                    os.rename(f'{temp}/{Pic}', f'{temp}/{nombreFinal}')
                elif str("."+Pic.split(".")[1])==".JPG":
                    nombreFinal = f'{str_Planta}_{str_CT}_{"V" + str(carpeta).zfill(2)}_{nombreImagen+"_T"}_{fechaHora}.JPG'
                    # temp = f'{dir0}/{str_Planta}/T/{str_CT}/{"V" + str(carpeta).zfill(2)}'
                    temp = f'{dir0}/{str_CT}/{"V" + str(carpeta).zfill(2)}'
                    os.makedirs(temp, exist_ok=True)
                    shutil.copy(Picdir,
                                temp)  # Hace copia de 'D:/ProcesamientoEnel_2023-10/Imagenes/Plantas/VDS/NCU20/100MEDIA/DJI_0651_T.JPG' ->
                    os.rename(f'{temp}/{Pic}', f'{temp}/{nombreFinal}')
                fechaHora_anterior = fechaHora
    return carpeta


# Modo según orden de carpetas
Orden = "CAMPO_VUELO"
# Agregar otro modo

# Source DE FOTOS ORDENADAS POR CT
root = tk.Tk()
root.withdraw()
if Orden == "CAMPO_VUELO":
    niveles = 2
    list_niveles = ["", "v"]
    dir0 = filedialog.askdirectory(title='Seleccione directorio con imágenes de un campo, separadas por vuelo')

else:
    dir0 = filedialog.askdirectory(title='Seleccione directorio de imágenes ordenadas por CT')
    niveles = 3
    list_niveles = ["", "", "v"]

# for nivel in range(niveles): # de 0 a niveles-1

iterador = 1
print(dir0)
for dir1 in os.listdir(dir0):
    print("Directorio de origen: "+dir1)
    if Orden == "CAMPO_VUELO":
        str_path = '/'.join(dir0.split('/')[:-2])  # 'D/:ProcesamientoEnel_2023-10/Imagenes/Plantas'
        str_Planta = dir0.split('/')[-3]  # 'VDS'
        str_CT = dir0.split('/')[-1]  # 'NCU20'
        str_V = dir1  # "105MEDIA"
        im_path = dir0 + '/' + dir1  # D:/ProcesamientoEnel_2023-10/Imagenes/Plantas/VDS/NCU20/105MEDIA
        # PlantaDir = join(dir0.split('/'))[:-1]
        # CTdir = dir0
        # Vdir = dir0 + '/' + dir1
        # for Pic in os.listdir(Vdir):
        if(str_V != "desktop.ini"):
            iterador_temp = copiar_imagenes(str_path, str_Planta, str_CT, str_V, im_path,iterador)
            iterador = iterador_temp + 1


    else:
        CTdir = dir0 + '/' + dir1