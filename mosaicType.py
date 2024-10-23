import cv2
import numpy as np


def GaussianBlur(image_path, boxes, ksize=(25, 25)):
    image = cv2.imread(image_path)

    for box in boxes:
        box = np.int32(box).reshape((-1, 2))

        x_min, y_min = np.min(box[:, 0]), np.min(box[:, 1])
        x_max, y_max = np.max(box[:, 0]), np.max(box[:, 1])

        roi = image[y_min:y_max, x_min:x_max]

        blurred_roi = cv2.GaussianBlur(roi, ksize, 0)

        image[y_min:y_max, x_min:x_max] = blurred_roi
    return image

def Pixelate(image_path, boxes, pixel_size=(10, 10)):
    image = cv2.imread(image_path)

    for box in boxes:
        box = np.int32(box).reshape((-1, 2))

        x_min, y_min = np.min(box[:, 0]), np.min(box[:, 1])
        x_max, y_max = np.max(box[:, 0]), np.max(box[:, 1])

        roi = image[y_min:y_max, x_min:x_max]

        roi_h, roi_w = roi.shape[:2]

        temp = cv2.resize(roi, pixel_size, interpolation=cv2.INTER_LINEAR)

        pixelated_roi = cv2.resize(temp, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)

        image[y_min:y_max, x_min:x_max] = pixelated_roi

    return image

def ColorBlock(image_path, boxes, color=(0, 0, 0)):
    image = cv2.imread(image_path)

    for box in boxes:
        box = np.int32(box).reshape((-1, 2))

        x_min, y_min = np.min(box[:, 0]), np.min(box[:, 1])
        x_max, y_max = np.max(box[:, 0]), np.max(box[:, 1])

        image[y_min:y_max, x_min:x_max] = color

    return image
