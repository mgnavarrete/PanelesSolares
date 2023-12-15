import albumentations as A
import numpy as np
from cv2 import INTER_CUBIC

def yolo_2_album(x_center, y_center, width, height):
    '''
    Transform Yolo annotation to albumentation annotation
    :param x_center: normalized x-coordinate of bbox center
    :param y_center: normalized y-coordinate of bbox center
    :param width: normalized width of bbox
    :param height: normalized height of bbox
    :return: (x_min, y_min), (x_max, y_max)
             normalized coordinates of upper left corner and lower right corner of bbox
    '''

    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2

    return (x_min, y_min), (x_max, y_max)

def album_2_yolo(x_min, y_min, x_max, y_max):
    '''
    Transform albumentation annotation to Yolo  annotation
    '''

    height = y_max - y_min
    width = x_max - x_min
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2

    return x_center, y_center, width, height


def yolo_2_plot(boxes, labels, colors=None):
    '''
    Convert list of dicts with annotation in yolo format to plot format
    :param boxes: list of dicts with annotations in yolo format, dict with the following keys: x, y, widht, height,
    obj_class, confidence
    :param labels: list of classes in order predefined in annotation
    :param colors: list of colors for any class
    :return: list with dicts with plot format, dict with following keys: point1, point2, label, color
    '''
    if colors is None:
        colors = [(0, 255, 0) for _ in range(len(labels))]
    format_boxes = list()
    for bbox in boxes:
        point1, point2 = yolo_2_album(bbox['x'], bbox['y'], bbox['width'], bbox['height'])
        confidence = bbox['confidence'] if 'confidence' in bbox.keys() else None
        new_bbox = {'point1': point1, 'point2': point2, 'label': labels[bbox['obj_class']],
                    'color': colors[bbox['obj_class']], 'confidence': confidence}
        format_boxes.append(new_bbox)
    return format_boxes


def instence_data_augmentation_object(format='yolo'):
    '''
    Function to instance pipeline for data augmentation in object detection problem, this pipeline is based on
    albumentation library.  The probability of image not altered is 0.5
    :param format: annotation format, default:yolo
    :return: object to tranform data
    '''
    transform = A.Compose(
        [
            A.OneOf([
                A.Sequential([
                    A.OneOf([
                        A.HorizontalFlip(),
                        A.VerticalFlip(),
                    ], p=1),
                    A.Rotate(limit=(-5, 5), p=0.5, interpolation=INTER_CUBIC),
                ]),
                A.OneOf([
                    A.Rotate(limit=(-5, 5), interpolation=INTER_CUBIC),
                    A.Rotate(limit=(175, 185), interpolation=INTER_CUBIC),
                ])
            ], p=0.375),
            A.Perspective(scale=0.05, p=0.0196),
            A.OneOf([
                A.GaussianBlur(),
                A.MotionBlur(),
                A.Lambda(image=lambda x, **kwargs: x + np.random.normal(0, np.random.rand(), x.shape),
                         bbox=lambda x, **kwargs: x),
            ], p=0.184),
        ],
        bbox_params=A.BboxParams(format=format, label_fields=['labels_idx']),
    )
    return transform


def instence_data_augmentation_object2(format='yolo'):
    '''
    Function to instance pipeline for data augmentation in object detection problem, this pipeline is based on
    albumentation library.  The probability of image not altered is 0.5
    :param format: annotation format, default:yolo
    :return: object to tranform data
    '''
    transform = A.Compose(
        [A.Rotate(limit=(-90,-90), p=1)],
        bbox_params=A.BboxParams(format=format, label_fields=['labels_idx']),
    )
    return transform