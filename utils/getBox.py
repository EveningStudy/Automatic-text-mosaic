def getTextbox(image,box, padding):
    x1, y1, x2, y2 = int(box[0]), int(box[2]), int(box[1]), int(box[3])

    padding = padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.shape[1], x2 + padding)
    y2 = min(image.shape[0], y2 + padding)

    roi = image[y1:y2, x1:x2]
    roi_height, roi_width = roi.shape[:2]
    ma_size = max(2, min(roi_width, roi_height) // 10)

    return x1, y1, x2, y2, roi_height, roi_width, ma_size, roi
