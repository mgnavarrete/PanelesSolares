from adentu_solar.utils import get_thermo_by_flir_extractor, to_uint8
from glob import glob
import fnv.file
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import rescale
import pandas
from datetime import datetime

fallas_dict = {'BB': 'Bus Bar', 'CC': 'Celula Caliente', 'MC': 'Modulo Corto Cirtuito', 'SCA': 'String Desconectado'}
fallas_id = {'BB': 4, 'CC': 9, 'MC': 5, 'SCA': 1}


class AdentuFlirPedestre:
    '''Clase que almacena las funcionalidades para el procesamiento de las imagenes pedestres tomadas con la FLIR One'''

    def __init__(self, path_pedestres, csv_path, ar_id=101, campos='C45-46'):
        self.csv_path = csv_path
        self.path_pedestres = path_pedestres
        self.point_text_template = ' {:.1f}'
        self.logo = self.logo()
        self.ar_id = ar_id
        self.campos = campos

    def logo(self):
        im = cv2.imread('adentu_flir.png')
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[7:-2, 4:-4]
        mask = (im < 200)
        im = mask * 255
        im = rescale(im, 1. / 5)
        return 255 * im / im.max()

    def add_logo(self, im, f=0.3):
        ls = self.logo.shape
        _mask = 1.0 * (self.logo > 5)[..., np.newaxis]

        im[:ls[0], :ls[1]] = (1 - f) * im[:ls[0], :ls[1]] * _mask + f * self.logo[..., np.newaxis] * _mask + (
                1 - _mask) * im[:ls[0], :ls[1]]

        return im

    def procesar_flir(self, path, panel=None, falla=None):
        im, meta = get_thermo_by_flir_extractor(path, metadata=True)
        im_aux = fnv.file.ImagerFile(path)
        im_aux.get_frame(0)
        assert len(im_aux.rois) == 3
        im_col = to_uint8(im.copy())
        rois = [x for x in im_aux.rois]
        temps = []
        for roi in rois[1:3]:
            cp_ = roi.center_position
            y = int(cp_['x'])
            x = int(cp_['y'])
            cv2.putText(im_col, self.point_text_template.format(im[x, y]), (y, x), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0,
                                                                                                                   0))
            im_col = cv2.circle(im_col, (y, x), 5, (0, 0, 0), 1)
            im_col = cv2.circle(im_col, (y, x), 2, (0, 0, 0), -1)
            temps.append(im[x, y])

        im_col = self.add_logo(im_col)

        if panel is not None and falla is not None:
            aux = im_col.copy()
            cv2.putText(aux, f'Tracker: {panel.split("-")[0]}', (200, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255,
                                                                                                           255))
            cv2.putText(aux, f'Modulo: {panel.split("-")[1]}', (200, 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255,
                                                                                                          255))
            cv2.putText(aux, f'Panel: {panel.split("-")[2]}', (200, 60), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255,
                                                                                                         255))
            cv2.putText(aux, falla, (200, 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
            im_col = 0.7 * im_col + 0.3 * aux

        return np.uint8(im_col), temps

    def procesar(self, path_save=None):
        now = datetime.now()
        consultas_path = f'./procesado_{now.day}del{now.month}del{now.year}_{now.hour}{now.minute}{now.second}.txt'

        with open(consultas_path, 'w') as f:
            pass

        if path_save is None:
            path_save = f'./procesado_{now.day}del{now.month}del{now.year}_{now.hour}{now.minute}{now.second}'

        # vemos si existe la carpeta, de lo contrario se crea
        if not os.path.isdir(path_save):
            os.mkdir(path_save)

        # Cargamos el csv
        data = pandas.read_csv(self.csv_path)

        # obtenemos path imagenes
        imagenes = glob(self.path_pedestres + '/**.jpg')
        imagenes = sorted(imagenes, key=lambda x: imagenes[0].split('\\')[-1].replace('.jpg', ''))

        for _, row in data.iterrows():
            panel = row['Panel'].replace('_', '-')
            panel_real = row['Panel'].replace('_', '-')
            if row['Si el panel esta correcto'] == True:
                segmentos = panel_real.split('-')
                segmentos[2] = str(int(row['Obs Panel']))
                panel_real = '-'.join(segmentos)
            tipo_falla = row['Obs Falla'] if row['Si la falla esta correcta'] else row['Tipo de falla']

            # im_path = imagenes[int(row['Correlativo Foto'])]
            im_path = imagenes[1]

            imf, temps = fp.procesar_flir(im_path, panel=panel_real, falla=fallas_dict[tipo_falla])

            if not pandas.isna(row['Codigo']):

                temps = sorted(temps, reverse=True)

                delta = round(temps[0] - temps[-1], 3)

                tipo_falla = fallas_id[tipo_falla]

                barcode = int(row['Codigo'])

                im_name = panel_real.replace('-', '_') + '.jpg'
                cv2.imwrite(f'{path_save}/{im_name}', imf)
                if len(temps) == 2:
                    p1, p2 = temps
                    consulta = f"insert into pedestre(ar_id, campos, panel, panel_real, barcode, im_name, tipo_falla, p1, p2, delta) values " \
                               f"({self.ar_id}, '{self.campos}', '{panel}', '{panel_real}', '{barcode}', '{im_name}'," \
                               f" {tipo_falla}, {p1}, {p2}, {delta});"
                else:
                    raise Exception('Ooooh Noooooo Voy a morir')

                with open(consultas_path, 'a') as f:
                    f.write(consulta + '\n')
            break
        print('Done')

if __name__ == '__main__':

    # lista = glob('./CDO 05-07-2022 Mañana - Tarde (FlirOne)/**.jpg')

    fp = AdentuFlirPedestre(path_pedestres='./CDO 05-07-2022 Mañana - Tarde (FlirOne)', csv_path='./simulado.csv')
    fp.procesar()
    # im, temps = fp.procesar_flir(lista[0], panel='1351_07_30', falla='Bus Bar')
    #
    # print(temps)
    # plt.figure()
    # plt.imshow(im[:, :, ::-1])
    # plt.show()
