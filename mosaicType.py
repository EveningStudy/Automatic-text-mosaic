import cv2
import numpy as np


def common(image_path, boxes, ksize=(25, 25)):
    """
    对检测到的文本区域进行模糊处理

    Parameters:
    - image_path: str, 输入图像的路径
    - boxes: list, 每个元素是检测到的文本区域的坐标（四边形）
    - ksize: tuple, 模糊核大小，决定模糊程度，默认 (25, 25)

    Returns:
    - image: 已经对文字区域进行模糊处理的图像
    """
    # 读取图像
    image = cv2.imread(image_path)

    # 对每个检测到的文字区域进行处理
    for box in boxes:
        box = np.int0(box).reshape((-1, 2))  # 将坐标转换为整型并重塑为2D坐标

        # 获取最小和最大坐标（用于确定矩形边界）
        x_min, y_min = np.min(box[:, 0]), np.min(box[:, 1])
        x_max, y_max = np.max(box[:, 0]), np.max(box[:, 1])

        # 取出该区域
        roi = image[y_min:y_max, x_min:x_max]

        # 对该区域进行高斯模糊处理
        blurred_roi = cv2.GaussianBlur(roi, ksize, 0)

        # 将模糊后的区域放回图像中
        image[y_min:y_max, x_min:x_max] = blurred_roi
    return image
