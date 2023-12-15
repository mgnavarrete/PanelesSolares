from cv2 import rectangle, getTextSize, FONT_HERSHEY_SIMPLEX, putText


def add_bbox(img, point1, point2, label=None, confidence=None, color=(0, 255, 0), thickness=3):
    '''
    Function to add bbox to image
    :param img: array image in format BGR
    :param point1: upper left point of bbox
    :param point2: bottom right point of bbox
    :param label: string label-class
    :param confidence: confidence of classification
    :param color: BGR tuple color for bbox
    :param thickness: thickness of bbox
    :return: img: image with bbox
    '''
    m, n = img.shape[0:2]
    point1 = (int(point1[0]*n), int(point1[1]*m))
    point2 = (int(point2[0]*n), int(point2[1]*m))
    img = rectangle(img, point1, point2, color, thickness)
    if label is not None or confidence is not None:
        text = ''
        if label is not None:
            text += f'{label}'
        if confidence is not None:
            text += f' {100 * confidence:.1f}%'
        (width_text, height_text), _ = getTextSize(text, FONT_HERSHEY_SIMPLEX, 0.5, 2)
        x1, y1 = point1
        img = rectangle(img, (x1 - 1, y1 - height_text - 1), (x1 + width_text, y1), color, -1)
        img = putText(img, text, (x1, y1), FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return img


def add_all_bboxes(img, boxes, thickness=3):
    '''
    Add all bboxes to image
    :param img: array image in format BGR
    :param boxes: list of dict with the following keys: point1, point2, color, confidence
    :param thickness: thickness of bbox
    :return: img: image with bboxes
    '''
    for bbox in boxes:
        img = add_bbox(img, bbox['point1'], bbox['point2'], label=bbox['label'], color=bbox['color'],
                       confidence=bbox['confidence'], thickness=thickness)
    return img
