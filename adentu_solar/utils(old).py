import flirimageextractor
import matplotlib.pyplot as plt
import PIL.Image
import subprocess
import PIL.Image
import numpy as np
import io
import json
from matplotlib import cm
from cv2 import LUT
import lensfunpy
import cv2


EXIFTOOL = "./exiftool"  # command to use exiftool


def get_thermo_by_flir_extractor(f, save_name=None, show=False, metadata=False):
    '''
    Read thermo image using flirimageextractor package
    :param f: jpg thermal image
    :param save_name: Name to save the file
    :param show: True for show thermal image
    :param metadata: True for extract metadata
    :return: temperature image in Celcius
    '''
    flir = flirimageextractor.FlirImageExtractor()
    try:
        flir.process_image(f)
        I = flirimageextractor.FlirImageExtractor.get_thermal_np(flir)
        I = I.astype(np.float32)
        if show:
            plt.figure()
            plt.imshow(I)
        if save_name is not None:
            plt.imsave(save_name + '.jpg', I, cmap='gray')
        if metadata:
            meta = get_string_meta_data(f)
            return I, meta
        return I
    except Exception as ex:
        print(ex, '\nFile is not a thermal image')


def get_temp_from_flir(path, radiometric_corr=True):
    '''
    Read data from JPG thermal image using exiftool
    :param path: path to jpg thermal image
    :param radiometric_corr: True for use radiometric correction
    :return: temperature (in Celcius) image, metadata
    '''
    # read meta-data and binary image
    meta = get_string_meta_data(path)
    img = get_raw_thermal_image(path)
    # converison to temperature:
    # according to http://u88.n24.queensu.ca/exiftool/forum/index.php/topic,4898.msg23972.html#msg23972
    # extract radiometric parameters
    r1, r2, b, f, o = tuple(meta["Planck{}".format(s)]
                            for s in ("R1", "R2", "B", "F", "O"))
    # for the case of emissivity != 1
    if radiometric_corr:
        emissivity = meta["Emissivity"]
        # drop the C on the temp and increase to Kelvin
        t_refl = float(meta["ReflectedApparentTemperature"][:-1]) + 273.15
        raw_refl = r1 / (r2 * (np.exp(b / t_refl) - f)) - o
        raw_obj = (img - (1 - emissivity) * raw_refl) / emissivity
        t = b / np.log(r1 / (r2 * (raw_obj + o)) + f) - 273.15
    # (for the case: emissivity == 1)
    else:
        t = b / np.log(r1 / (r2 * (img + o)) + f)

    return t, meta


def get_raw_thermal_image(path):
    '''
    Use exiftool to extract 'RawThermalImage' from FLIR-JPG
    :param path: path to file
    :return: array thermal data
    '''
    # call exiftool and extract binary data
    cmd = [EXIFTOOL, path, "-b", "-RawThermalImage"]
    # cmd = [EXIFTOOL, path, "-b", "-EmbeddedImage"]
    r_data = subprocess.check_output(cmd)
    # read in image (should detect image format)
    im = PIL.Image.open(io.BytesIO(r_data))
    # convert image to array
    return np.array(im)


def get_string_meta_data(path):
    '''
    Read metadata exif-data using exiftool
    :param path: path to thermal image file
    :return: metadata
    '''
    # call exiftool with 'JSON'-output flag
    cmd = [EXIFTOOL, path, "-a", "-j", "-z"]
    dta = subprocess.check_output(cmd, universal_newlines=True)
    # convert to stream and load using 'json' library
    data = json.load(io.StringIO(dta))
    # reduce dimension if singleton
    if isinstance(data, list) and len(data) == 1:
        data = data[0]
    return data


def adjust_gamma(img, gamma=1.0):
    '''
    Gamma correction using LUT
    :param img: array image with dtype uint8
    :param gamma:gamma factor for correction
    :return: img: image corrected
    '''
    # build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return LUT(img, table)


def to_uint8(img, gamma=None):
    '''
    Normalize image to uint8 with inferno cmap to plot in BGR mode
    :param img: array image
    :param gamma: gamma factor for gamma correction
    :return: img: array image dtype uint8
    '''
    img = np.float32(img)
    img -= np.min(img)
    img /= np.max(img)
    img = np.uint8(255 * cm.inferno(img))[..., :3]
    img = img[..., [2, 1, 0]].copy()
    if gamma is not None:
        img = adjust_gamma(img, gamma=gamma)
    return img

def to_uint8g(img, gamma=None):
    '''
    Normalize image to uint8 with inferno cmap to plot in BGR mode
    :param img: array image
    :param gamma: gamma factor for gamma correction
    :return: img: array image dtype uint8
    '''
    img = np.float32(img)
    img -= np.min(img)
    img /= np.max(img)
    img = np.uint8(255 * cm.gray(img))[..., :3]
    img = img[..., [2, 1, 0]].copy()
    if gamma is not None:
        img = adjust_gamma(img, gamma=gamma)
    return img

def gen_color_pallete(n_classes):
    '''
    Generate n random colors in a list
    :param n_classes: number of classes/labels ie: number of colors needed
    :return: list of colors, any color is a tuple in RGB
    '''
    color = np.random.randint(0, 255, (n_classes, 3))
    color = [color[i].tolist() for i in range(n_classes)]
    return color


def undistort_zh20t(im):
    st = '''
    <lensdatabase version="1">

        <mount>
            <name>Pentax K</name>
            <compat>M42</compat>
        </mount>

        <lens>
            <maker>Pentax</maker>
            <model>SMC Pentax M 13.5mm f/1.0</model>
            <mount>Pentax K</mount>
            <cropfactor>1.0</cropfactor>
            <focal value="13.5" />
            <aperture min="1.0" max="1.0" />
            <type>rectilinear</type>
            <calibration>
                <!-- WARNING: this calibration data is completely bogus :) -->
                <distortion model="ptlens" focal="13.5" a="0.01865" b="-0.06932" c="0.05956" />
            </calibration>
        </lens>

        <camera>
            <maker>Pentax</maker>
            <model>Pentax K10D</model>
            <mount>Pentax KAF2</mount>
            <cropfactor>1.0</cropfactor>
        </camera>

    </lensdatabase>'''

    cam_maker = 'Pentax'
    cam_model = 'Pentax K10D'
    lens_maker = 'Pentax'
    lens_model = 'SMC Pentax M 13.5mm f/1.0'

    db = lensfunpy.Database(xml=st)
    cam = db.find_cameras(cam_maker, cam_model)[0]
    lens = db.find_lenses(cam, lens_maker, lens_model)[0]
    focal_length = 13.5
    aperture = 1
    distance = 5

    height, width = im.shape[0], im.shape[1]

    mod = lensfunpy.Modifier(lens, cam.crop_factor, width, height)
    mod.initialize(focal_length, aperture, distance)

    undist_coords = mod.apply_geometry_distortion()
    im_undistorted = cv2.remap(im, undist_coords, None, cv2.INTER_LANCZOS4)
    return im_undistorted

def undistort_m2ea_th(im):
    st = '''
    <lensdatabase version="1">

        <mount>
            <name>Pentax K</name>
            <compat>M42</compat>
        </mount>

        <lens>
            <maker>Pentax</maker>
            <model>SMC Pentax M 13.5mm f/1.0</model>
            <mount>Pentax K</mount>
            <cropfactor>1.0</cropfactor>
            <focal value="13.5" />
            <aperture min="1.0" max="1.0" />
            <type>rectilinear</type>
            <calibration>
                <!-- WARNING: this calibration data is completely bogus :) -->
                <distortion model="ptlens" focal="9" a="0.02356" b="-0.13063" c="0.13631" />
            </calibration>
        </lens>

        <camera>
            <maker>Pentax</maker>
            <model>Pentax K10D</model>
            <mount>Pentax KAF2</mount>
            <cropfactor>1.0</cropfactor>
        </camera>

    </lensdatabase>'''

    cam_maker = 'Pentax'
    cam_model = 'Pentax K10D'
    lens_maker = 'Pentax'
    lens_model = 'SMC Pentax M 13.5mm f/1.0'

    db = lensfunpy.Database(xml=st)
    cam = db.find_cameras(cam_maker, cam_model)[0]
    lens = db.find_lenses(cam, lens_maker, lens_model)[0]
    focal_length = 9
    aperture = 1
    distance = 5

    height, width = im.shape[0], im.shape[1]

    mod = lensfunpy.Modifier(lens, cam.crop_factor, width, height)
    mod.initialize(focal_length, aperture, distance)

    undist_coords = mod.apply_geometry_distortion()
    im_undistorted = cv2.remap(im, undist_coords, None, cv2.INTER_LANCZOS4)
    return im_undistorted



def get_thermal_zh20t(path):
    input_file = 'test.raw'
    cmd = ['./dji_thermal_sdk_v1.4_20220929/utility/bin/windows/release_x86/dji_irp.exe', '-s', path, '-a', 'measure', '-o', f'{input_file}', '--measurefmt', 'float32']
    subprocess.call(cmd)
    npimg = np.fromfile(input_file, dtype=np.float32)
    imageSize = (512, 640)
    npimg = npimg.reshape(imageSize)
    return npimg


def get_thermal_zh20t2(n):
    def get_thermal_zh20t(path):
        input_file = f'test_{n}.raw'
        cmd = ['./dji_thermal_sdk_v1.4_20220929/utility/bin/windows/release_x86/dji_irp.exe', '-s', path, '-a', 'measure', '-o', f'{input_file}', '--measurefmt', 'float32']
        subprocess.call(cmd)
        npimg = np.fromfile(input_file, dtype=np.float32)
        imageSize = (512, 640)
        npimg = npimg.reshape(imageSize)
        return npimg
    return get_thermal_zh20t


def undistort_mv2en(im):
    st = '''
    <lensdatabase version="1">

        <mount>
            <name>Pentax K</name>
            <compat>M42</compat>
        </mount>

        <lens>
            <maker>Pentax</maker>
            <model>SMC Pentax M 13.5mm f/1.0</model>
            <mount>Pentax K</mount>
            <cropfactor>1.0</cropfactor>
            <focal value="13.5" />
            <aperture min="1.0" max="1.0" />
            <type>rectilinear</type>
            <calibration>
                <!-- WARNING: this calibration data is completely bogus :) -->
                <distortion model="ptlens" focal="38" a="-0.015214" b="0.00604" c="-0.029068" />
            </calibration>
        </lens>

        <camera>
            <maker>Pentax</maker>
            <model>Pentax K10D</model>
            <mount>Pentax KAF2</mount>
            <cropfactor>1.0</cropfactor>
        </camera>

    </lensdatabase>'''

    cam_maker = 'Pentax'
    cam_model = 'Pentax K10D'
    lens_maker = 'Pentax'
    lens_model = 'SMC Pentax M 13.5mm f/1.0'

    db = lensfunpy.Database(xml=st)
    cam = db.find_cameras(cam_maker, cam_model)[0]
    lens = db.find_lenses(cam, lens_maker, lens_model)[0]
    focal_length = 38
    aperture = 2.8
    distance = 5

    height, width = im.shape[0], im.shape[1]

    mod = lensfunpy.Modifier(lens, cam.crop_factor, width, height)
    mod.initialize(focal_length, aperture, distance)

    undist_coords = mod.apply_geometry_distortion()
    im_undistorted = cv2.remap(im, undist_coords, None, cv2.INTER_LANCZOS4)
    return im_undistorted