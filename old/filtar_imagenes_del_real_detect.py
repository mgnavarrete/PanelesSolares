import os
import shutil
import json
import cv2

cardinal = cv2.imread('./punto-cardinal.png')
mask = cardinal > 0


def add(impath, angle):
    im = cv2.imread(impath)
    if -99 < angle < -81:
        pass
    elif 81 < angle < 99:
        print('Rotado')
        im = cv2.rotate(im, cv2.ROTATE_180)
    im[:24, -24:, :] = im[:24, -24:, :] * (1 - mask) + mask * cardinal
    cv2.imwrite(impath, im)

# cambiar path
#path = './reEtiquetasPLS/re_etiquetar_CP/'
path = 'E:/ProcesamientoEnel_2023-04/Imagenes/Domeyko/DMK/Z5/PV025PP/'

if not os.path.exists(path + 'para_subir'):
    os.mkdir(path + 'para_subir')
else:
    shutil.rmtree(path + 'para_subir')
    os.mkdir(path + 'para_subir')

with open(path + 'real_detect.json') as f:
    poligonos = json.load(f)

for p in poligonos:
    name = p['name']
    shutil.copy(path + 'detect/' + name, path + 'para_subir/' + name)
    str_corr = name.split('_T_')[1].split('_')[0]
    real_name = name.split('_T_')[0] + '_T_' + str_corr + '.txt'
    with open(path+'metadata/'+real_name, 'r') as f:
        data = json.load(f)
    angle = float((data['GimbalYawDegree']))
    add(path + 'para_subir/' + name, angle)

print('Done')
