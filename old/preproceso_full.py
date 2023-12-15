import glob
import cv2
import numpy as np
from adentu_solar.utils import to_uint8, undistort_zh20t, get_thermal_zh20t, undistort_m2ea_th

np.random.seed(10)

'''
Pre-proceso para subir imagenes a CVAT 
Este escript se ejecuta en consola con el argumento path ej:
python3 preproceso_cvst.py --path my_path

el path tiene es a una carpeta de una zona, la cual debe contener carpetas de los vuelos y detro de estas las imagenes

'''

if __name__ == '__main__':
    import argparse
    import os
    import tkinter as tk
    from tkinter import filedialog

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="path donde se encuentran las imagenes")

    args = parser.parse_args()
    if args.path is not None:
        path_save = args.path + 'P'

        if not os.path.exists(path_save):
            os.mkdir(path_save)
        lista = glob.glob(args.path + '/**/*.JPG', recursive=True)
        lista = [l.replace('\\', '/') for l in lista]

        for l in lista:
            print(l)
            im = get_thermal_zh20t(l)
            im = to_uint8(im)
            #im = undistort_zh20t(im)
            im = undistort_m2ea_th(im)
            splits = l.split('/')
            if splits[-2] == 'T':
                name = 'v' + splits[-3].split(' ')[-1] + '_' + splits[-1]
            else:
                name = 'v' + splits[-2].split(' ')[-1] + '_' + splits[-1]
            cv2.imwrite(path_save + "/" + name, im)

    else:
        root = tk.Tk()
        root.withdraw()
        lista = filedialog.askopenfilenames(title='Seleccione las imagenes')
        path_save = filedialog.askdirectory(title='Seleccione path de guardado')

        for l in lista:
            print(l)
            im = get_thermal_zh20t(l)
            im = to_uint8(im)
            im = undistort_zh20t(im)
            splits = l.split('/')
            name = splits[-1]
            if cv2.imwrite(path_save + '/' + name, im):
                print("Imagen guardada: " + name)
            else:
                print("ERROR al guardar imagen: " + name)