import cv2
from utils import detect
from utils import getBox


def common(image, padding, lang):
    h_list, v_list = detect.detectBox(image, lang)
    for box in h_list:
        x1, y1, x2, y2, roi_height, roi_width, ma_size, roi = getBox.getTextbox(image, box, padding)

        small_roi = cv2.resize(roi, (ma_size, ma_size), interpolation=cv2.INTER_LINEAR)
        ma_roi = cv2.resize(small_roi, (roi_width, roi_height), interpolation=cv2.INTER_NEAREST)
        image[y1:y2, x1:x2] = ma_roi
    return image
