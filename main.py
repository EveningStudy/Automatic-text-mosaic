import easyocr
import cv2


reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)
image_path = '111.png'
image = cv2.imread(image_path)

results = reader.readtext(image)


for (box, a, b) in results:
    top_left = (int(box[0][0]), int(box[0][1]))
    bottom_right = (int(box[2][0]), int(box[2][1]))

    padding = 5
    top_left = (max(0, top_left[0] - padding), max(0, top_left[1] - padding))
    bottom_right = (min(image.shape[1], bottom_right[0] + padding), min(image.shape[0], bottom_right[1] + padding))


    roi = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    roi_height, roi_width = roi.shape[:2]
    ma_size = max(2, min(roi_width, roi_height) // 10)
    small_roi = cv2.resize(roi, (ma_size, ma_size), interpolation=cv2.INTER_LINEAR)
    ma_roi = cv2.resize(small_roi, (roi_width, roi_height), interpolation=cv2.INTER_NEAREST)

    image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = ma_roi





cv2.imwrite('111_ma.png', image)


