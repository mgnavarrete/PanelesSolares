from adentu_solar.utils import get_string_meta_data, undistort_zh20t,get_thermal_zh20t
from adentu_solar.utils import to_uint8g, gen_color_pallete
from adentu_solar.transform import yolo_2_plot, yolo_2_album
from adentu_solar.display import add_all_bboxes
import subprocess
import numpy as np
import json
import cv2
import utm
import string
from osgeo import ogr
import tkinter as tk
from tkinter import filedialog
import os
import simplekml
np.random.seed(10)
from glob import glob
import ast

def get_thermal(path):
    cmd = ['dji_irp.exe', '-s', path, '-a', 'measure', '-o', 'test.raw', '--measurefmt', 'float32']
    subprocess.call(cmd)
    input_file = 'test.raw'
    npimg = np.fromfile(input_file, dtype=np.float32)
    imageSize = (512, 640)
    npimg = npimg.reshape(imageSize)
    return npimg

def dms2dd(data):
    dd = float(data[0]) + float(data[1]) / 60 + float(data[2]) / (60 * 60)
    if data[3] == 'W' or data[3] == 'S':
        dd *= -1
    return dd

def save_georef_matriz(data,desp_este=0,desp_norte=0,desp_yaw=0):

    lat = data['GPSLatitude'].replace('\'', '').replace('"', '').split(' ')
    lng = data['GPSLongitude'].replace('\'', '').replace('"', '').split(' ')
    img_height = int(data['ImageHeight'])
    img_width = int(data['ImageWidth'])
    tamano_pix = 0.000012
    dis_focal = float(data['FocalLength'][:-2]) / 1000
    yaw = np.pi * (float(data["GimbalYawDegree"])+desp_yaw) / 180


    try: 
        distancia_laser = float(data["LRFTargetDistance"])
        lat_laser = float(data["LRFTargetLat"])
        lon_laser = float(data["LRFTargetLon"])
        GSD = tamano_pix * distancia_laser / dis_focal
        center = utm.from_latlon(lat_laser, lon_laser)
        
    except:
        for v in lat:
            if v == 'deg':
                lat.pop(lat.index(v))

        for v in lng:
            if v == 'deg':
                lng.pop(lng.index(v))
        center = utm.from_latlon(dms2dd(lat), dms2dd(lng))
        GSD = tamano_pix * float(data['RelativeAltitude']) / dis_focal



   

    mid_width = img_width / 2

    Matriz_y = np.zeros((img_height, img_width))
    Matriz_x = np.zeros((img_height, img_width))

    for pixel_y in range(img_height):
        distancia_y = (pixel_y - img_height / 2 + 0.5) * GSD
        Matriz_y[pixel_y, :] = np.ones(img_width) * -1 * distancia_y

    matriz_gsd_y = (np.append(Matriz_y[:, 0], Matriz_y[-1, 0]) - np.append(Matriz_y[0, 0], Matriz_y[:, 0]))
    matriz_gsd_x = matriz_gsd_y[1:-1]  # asumimos pixeles cuadrados
    matriz_gsd_x = np.append(matriz_gsd_x[0], matriz_gsd_x[:])

    for pixel_y in range(img_height):
        gsd_x = matriz_gsd_x[pixel_y]
        distancia_x = -gsd_x * (mid_width - 0.5)
        for pixel_x in range(img_width):
            Matriz_x[pixel_y, pixel_x] = distancia_x
            distancia_x = distancia_x + gsd_x

    # AJUSTAR OFFSET DEL GPS, VALORES REFEREMCIALES 
    Matriz_Este = Matriz_y * np.sin(yaw) - Matriz_x * np.cos(yaw) + center[0] + desp_este
    Matriz_Norte = Matriz_y * np.cos(yaw) + Matriz_x * np.sin(yaw) + center[1] + desp_norte

    print(center[0], center[1])

    Matriz_zonas_1 = np.ones((img_height, img_width)) * center[2]
    Matriz_zonas_2 = np.ones((img_height, img_width)) * string.ascii_uppercase.find(center[3])

    matriz_puntos_utm = np.concatenate(
        [Matriz_Este[..., np.newaxis], Matriz_Norte[..., np.newaxis], Matriz_zonas_1[..., np.newaxis],
         Matriz_zonas_2[..., np.newaxis]], axis=-1)
    return matriz_puntos_utm

def inicializar():
    root = tk.Tk()
    root.withdraw()
    im_path = filedialog.askopenfilenames(title='Seleccione imagenes')
    path_labels = filedialog.askdirectory(title='Seleccione path etiquetas')
    path_obj = filedialog.askopenfilename(title='Seleccione obj.names')
    path_save = '/'.join(im_path[0].split('/')[:-1]) + 'P'
    if not os.path.exists(path_save):
        os.mkdir(path_save)

    if not os.path.exists(path_save + '/metadata'):
        os.mkdir(path_save + '/metadata')

    if not os.path.exists(path_save + '/detect'):
        os.mkdir(path_save + '/detect')

    if not os.path.exists(path_save + '/georef_numpy'):
        os.mkdir(path_save + '/georef_numpy')


    names = ["Tipo I - StringDesconectado",
             "Tipo II - StringCortoCircuito",
             "Tipo III - ModuloCircuitoAbierto",
             "Tipo IV - BusBar",
             "Tipo V - ModuloCortoCircuito",
             "Tipo VI - CelulaCaliente",
             "Tipo VII - ByPass",
             "Tipo VIII - PID",
             "Tipo IX - JunctionBoxCaliente"]
    return [im_path,path_labels,path_obj,path_save,names]
    
def detect(im_path,path_labels,path_obj,path_save,names,desp_este,desp_norte,desp_yaw):
    with open(path_obj) as f:
        names2 = f.readlines()
        names2 = [l.replace('\n', '') for l in names2]


    mapeo = {0:names.index(names2[0]), 1: names.index(names2[1]), 2:names.index(names2[2]),
             3:names.index(names2[3]), 4:names.index(names2[4]), 5:names.index(names2[5]),
             6:names.index(names2[6]), 7:names.index(names2[7]), 8:8}

    paleta = gen_color_pallete(len(names))
    paleta[3] = [255,0,0][::-1]
    paleta[4] = [255, 191, 0][::-1]
    paleta[2] = [128, 255, 0][::-1]
    paleta[0] = [0, 128, 255][::-1]
    paleta[5] = [255, 0, 255][::-1]

    paleta2 = paleta.copy()
    for i in range(len(paleta2)):
        paleta[i] = paleta2[mapeo[i]]

    bbox_path = [path_labels + '/' + l.split("/")[-1].replace('.JPG', '.txt') for l in im_path]

    i = 0
    for imp, bp in zip(im_path, bbox_path):

        file_path = path_save + f'/metadata/{imp.split("/")[-1].replace(".JPG", ".txt")}'

        try:
            try:
                file =open(file_path,'r')
                metadata=json.load(file)
                print("metadata_encontrada ",file_path)
                file.close()
            except: import d

         
        except:
            metadata = get_string_meta_data(imp)
            file = open(file_path, 'w')
            file.write(json.dumps(metadata, indent=4, sort_keys=True, default=str))
            file.close()

        matriz_geo = save_georef_matriz(metadata,desp_este,desp_norte,desp_yaw)
        geo_name = path_save + f'/georef_numpy/{imp.split("/")[-1].replace(".JPG", ".npy")}'
        np.save(geo_name, matriz_geo)

        im = get_thermal(imp)

        with open(bp, 'r') as f:
            bboxes = f.readlines()

        geo_matrix = matriz_geo
        geo_matrix_shape = geo_matrix.shape
        conf = 1
        name = imp.split('/')[-1]

        if bboxes.__len__() > 0:
            to_file_bbox = ''
            pdata = []
            tdata = []
            for _b in bboxes:
                cls, xx, yy, ww, hh = _b.split(" ")
                xx = float(xx)
                yy = float(yy)
                ww = float(ww)
                hh = float(hh)
                dd = (xx-0.5)**2 + (yy-0.5)**2

                # xx = 0.5
                # yy = 0.5
                # ww = 0.99999
                # hh = 0.99999

                pdata.append({'x': xx, 'y': yy, 'width': ww, 'height': hh, 'obj_class': int(cls)})

                c1, c2 = yolo_2_album(xx, yy, ww, hh)
                c1 = (int(c1[0]*640), int(c1[1]*512))
                c2 = (int(c2[0]*640), int(c2[1]*512))

                if c1[0] >= geo_matrix_shape[1]:
                    c1 = (geo_matrix_shape[1] - 1, c1[1])
                if c2[0] >= geo_matrix_shape[1]:
                    c2 = (geo_matrix_shape[1] - 1, c2[1])

                if c1[1] >= geo_matrix_shape[0]:
                    c1 = (c1[0], geo_matrix_shape[0] - 1)
                if c2[1] >= geo_matrix_shape[0]:
                    c2 = (c2[0], geo_matrix_shape[0] - 1)

                p1_utm = geo_matrix[c1[1]][c1[0]]
                p1_ll = utm.to_latlon(p1_utm[0], p1_utm[1], int(p1_utm[2]), string.ascii_uppercase[int(p1_utm[3])])

                p2_utm = geo_matrix[c1[1]][c2[0]]
                p2_ll = utm.to_latlon(p2_utm[0], p2_utm[1], int(p2_utm[2]), string.ascii_uppercase[int(p2_utm[3])])

                p3_utm = geo_matrix[c2[1]][c2[0]]
                p3_ll = utm.to_latlon(p3_utm[0], p3_utm[1], int(p3_utm[2]), string.ascii_uppercase[int(p3_utm[3])])

                p4_utm = geo_matrix[c2[1]][c1[0]]
                p4_ll = utm.to_latlon(p4_utm[0], p4_utm[1], int(p4_utm[2]), string.ascii_uppercase[int(p4_utm[3])])

                to_file_bbox += '{"poly" : "\'POLYGON(( %s %s, %s %s, %s %s, %s %s, %s %s ))\'","type" : "%s","conf" : "%s" ,' % (
                p1_ll[0], p1_ll[1], p2_ll[0], p2_ll[1], p3_ll[0], p3_ll[1], p4_ll[0], p4_ll[1], p1_ll[0], p1_ll[1],
                names[mapeo[int(cls)]], conf)
                to_file_bbox += '"geo_json" : {"type": "Polygon","coordinates":[[  [%s, %s],[%s, %s],[%s, %s],[%s, %s],[%s, %s] ]] }, ' % (
                    p1_ll[0], p1_ll[1], p2_ll[0], p2_ll[1], p3_ll[0], p3_ll[1], p4_ll[0], p4_ll[1], p1_ll[0], p1_ll[1])

                to_file_bbox += f'"name" : "{name}", "dd": {dd}' + '}'
                to_file_bbox = to_file_bbox.replace('}{', '},{')

            with open(path_save + f'/detect/{imp.split("/")[-1].replace(".JPG", ".json")}', 'w') as f:
                f.write('[')
                f.write(to_file_bbox)
                f.write(']')
            imi = to_uint8g(im)

            all_boxes = yolo_2_plot(pdata, [None for _ in range(len(names))], colors=paleta)
            imi = undistort_zh20t(imi)
            imi = add_all_bboxes(imi, all_boxes, 1)
            im_name = path_save + f'/detect/{imp.split("/")[-1]}'
            cv2.imwrite(im_name, imi)

def desplazamiento():
    while True:
        try:
            desp_norte = float(input('Introduzca el desplazamiento hacia el norte en metros')) 
            break
        except:
            continue
    while True:
        try:
            desp_este = float(input('Introduzca el desplazamiento hacia el este en metros'))
            break
        except:
            continue
    while True:
        try:
            desp_yaw = float(input('Introduzca la rotacion en Grados Sentido horario')) 
            break
        except:
            continue
    return [desp_este,desp_norte,desp_yaw]


def real_detect(path_save):
    polygons = []

    data_path = f'{path_save}/detect'
    write_path = path_save
    n_images = {}
    for file in os.listdir(data_path):
        ext = file.split('.')
        if ext[1] == 'json':
            with open(data_path + '/' + file) as json_file:
                data_file = json.load(json_file)
                for data in data_file:
                    polygons.append({"data": data, "poly": ogr.CreateGeometryFromJson(str(data['geo_json']))})
                    if data['name'] in n_images.keys():
                        n_images[data['name']] += 1
                    else:
                        n_images[data['name']] = 1

    for p in polygons:
        if p['data']['type'] == 'Tipo IX - JunctionBoxCaliente':
            polygons.pop(polygons.index(p))
    polygons.sort(reverse=True, key=lambda x: (n_images[x['data']['name']], 0.5 - x['data']['dd']))

    i = 0
    print('jiji', len(polygons))
    while i < len(polygons):
        j = i + 1
        while j < len(polygons):
            if polygons[i]["data"]["name"] != polygons[j]["data"]["name"] and polygons[i]["data"]["type"] == \
                    polygons[j]["data"]["type"]:
                dist = polygons[i]['poly'].Distance(polygons[j]['poly'])
                if dist < 5E-5:  #### <----- Umbral de distancia
                    polygons.pop(j)
                    continue
            j += 1
        i += 1
    print(len(polygons))
    read_detec = open(write_path + '/real_detect.json', 'w')
    data = []
    for i, poly in enumerate(polygons):
        poly["data"].__delitem__('dd')
        data.append(poly["data"])
    read_detec.write(json.dumps(data))
    read_detec.close()

    pipi = {}
    for f in polygons:
        if f['data']['name'] not in pipi.keys():
            pipi[f['data']['name']] = [f['data']['type']]
        else:
            pipi[f['data']['name']].append(f['data']['type'])
    t = 0
    for l in pipi.values():
        t += len(set(l))

def generar_kml_imagenes(desp_este,desp_norte,desp_yaw,path_save):

    apellido="DN"+str(desp_norte)+"DE"+str(desp_este)
    with open(path_save+apellido+'.kml', 'w') as file:
        a=f'''<?xml version="1.0" encoding="UTF-8"?>
    <kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2" xmlns:kml="http://www.opengis.net/kml/2.2" xmlns:atom="http://www.w3.org/2005/Atom">
    <Folder>
        <name>{apellido}</name>
        '''
        file.write(a)
        for f_name in glob(f"{path_save}/detect/*.JPG"):
            all_path=f_name
            nombre=os.path.basename(os.path.normpath(f_name)).replace(".JPG","")

            #with open(f"C:/Users/vicen/OneDrive/Escritorio/detect/{nombre}.txt")  as f1:
            #with open(f"C:/Users/vicen/OneDrive/Escritorio/Adentu/Solcito/finis/metadata/{nombre}.txt")  as f1:
            #with open(f"C:/Users/vicen/OneDrive/Escritorio/Adentu/Solcito/la_silla_pros/metadata/{nombre}.txt")  as f1:
            #with open(f"C:/Users/vicen/OneDrive/Escritorio/Adentu/Solcito/superposicion_de_imagenes/metadata T/{nombre}.txt")  as f1:
            with open(f"{path_save}/metadata/{nombre}.txt")  as f1:
                data2=json.load(f1)


            m = save_georef_matriz(data2,desp_este,desp_norte,desp_yaw)
            p1_ll = utm.to_latlon(m[0][0][0], m[0][0][1], int(m[0][0][2]), string.ascii_uppercase[int(m[0][0][3])])
            p2_ll = utm.to_latlon(m[0][-1][0], m[0][-1][1], int(m[0][-1][2]), string.ascii_uppercase[int(m[0][-1][3])])
            p3_ll = utm.to_latlon(m[-1][-1][0], m[-1][-1][1], int(m[-1][-1][2]), string.ascii_uppercase[int(m[-1][-1][3])])
            p4_ll = utm.to_latlon(m[-1][0][0], m[-1][0][1], int(m[-1][0][2]), string.ascii_uppercase[int(m[-1][0][3])])

            
            cordinates=f"{str(p4_ll[1])},{str(p4_ll[0])},0 {str(p3_ll[1])},{str(p3_ll[0])},0 {str(p2_ll[1])},{str(p2_ll[0])},0 {str(p1_ll[1])},{str(p1_ll[0])},0 "


            
            a=f'''<GroundOverlay>
            <name>{apellido+nombre}</name>
            <Icon>
                <href>{all_path}</href>
                <viewBoundScale>0.75</viewBoundScale>
            </Icon>
            <gx:LatLonQuad>
                <coordinates>
                    {cordinates} 
                </coordinates>
            </gx:LatLonQuad>
        </GroundOverlay>
        '''
            file.write(a)
        a='''</Folder>
    </kml>'''
        file.write(a)

def generar_kml_fallas(path_save):

    kml = simplekml.Kml()

    for f_name in glob(f'{path_save}/detect/*.json'):
        
        with open(f_name) as file:
            data = json.load(file)
            numero_de_falla=1
            for poligono in data:
                print(poligono["name"])
                
                print(poligono["type"])

                cordenadas=poligono["poly"]

                listcordenadas_2=[]

                listcordenadas=cordenadas.split()
                listcordenadas=listcordenadas[1:-1]
                for i in range(int(len(listcordenadas)/2)):
                    listcordenadas_2.append((float(listcordenadas[2*i+1].replace(",", "")),(float(listcordenadas[2*i].replace(",", "")))))

                print(listcordenadas_2)
                pol = kml.newpolygon(name=poligono["name"]+"_"+str(numero_de_falla),

                outerboundaryis=listcordenadas_2)
                # innerboundaryis=[(18.43348,-33.98985),(18.43387,-33.99004),(18.43410,-33.98972),
                #                       (18.43371,-33.98952),(18.43348,-33.98985)])
                
                pol.style.polystyle.color = simplekml.Color.red
                pol.style.polystyle.outline =0
                #pol.style.polystyle.fill = 1

                numero_de_falla=numero_de_falla+1

    kml.save("fallas.kml")

if __name__ == '__main__':
    
    [im_path,path_labels,path_obj,path_save,names]= inicializar()

    

    while True:
        seleccion = input('''
        
        
    Elija su opcion:
        0 Ayuda
        1 Procesar
        2 Generar Real detect
        3 Generar KML Imagenes
        4 Cambiar Directorio 
        5 Salir 
        
        
        Introduzca su opción:  ''')

        if(seleccion=="0"):
            print('''En este Script realiza las siguientes funcionalidades 
            
1 Procesar: 

    El script genera una carpeta con sufijo 'P'(nombre de carpeta igual al de las originales) para guardar los las
    detecciones, metadata, georef y json de detecciones. 

    Crea un KML con las imagenes ubicadas segun la metadata de las imagenes con el desplazamiento agregado
    
            
2 Generar Real detect:

    Genera un unico json con los poligonos filtrados ppor ubicacion e imagen para cargarlo a la plataforma

    Crea un KML de las Fallas sin desplazamiento a partir del detect se guarda en la misma carpeta en la que se corre el codigo 


3 Crear KML de Imagenes

    Crea un KML con las imagenes ubicadas segun la metadata de las imagenes con el desplazamiento agregado


4 Cambiar Directorio:

    Seleccionar imagenes a procesar, originales(no las del cvat, del cvat deben descargarse solo las etiquetas, de lo contrario genera un error al manejar grandes cantidades de imagenes)
    Seleccionar la carpeta de etiquedas descagadas, (en formato yolo la carpeta se llama obj_train_data)
    Seleccionar archivo obj.name descargado del cvat            
            ''')
 

        if(seleccion=="1"):
            print("Iniciando Procesamiento")
            [desp_este,desp_norte,desp_yaw]=desplazamiento()
            detect(im_path,path_labels,path_obj,path_save,names,desp_este,desp_norte,desp_yaw)
            print("Procesamiento Listo")
            print("Generarando KML imagenes")
            generar_kml_imagenes(desp_este,desp_norte,desp_yaw,path_save)
            print("KML imagenes Listo")
        
        if(seleccion=="2"):
            print("Iniciando REAL_DETECT")
            real_detect(path_save)
            print("REAL_DETECT Listo")
            print("Generarando KML fallas")
            generar_kml_fallas(path_save)
            print("KML fallas Listo")
        
        if(seleccion=="3"):
            print("Generarando KML imagenes")
            generar_kml_imagenes(desp_este,desp_norte,desp_yaw,path_save)
            print("KML imagenes Listo")

        if(seleccion=="4"):
            [im_path,path_labels,path_obj,path_save,names]= inicializar()

        if(seleccion=="5"):
            break

            


    


