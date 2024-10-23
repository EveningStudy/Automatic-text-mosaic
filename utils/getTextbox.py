import torch
import cv2
import numpy as np
from utils.craft import CRAFT
from utils.craft_utils import getDetBoxes, adjustResultCoordinates
from utils.detection import copyStateDict, resize_aspect_ratio, normalizeMeanVariance


# Initialize the CRAFT model
def load_craft_model(model_path, device='cpu', quantize=True):
    craft_net = CRAFT()  # Initialize the CRAFT model
    state_dict = torch.load(model_path, map_location=device)
    craft_net.load_state_dict(copyStateDict(state_dict))

    if device == 'cpu' and quantize:
        torch.quantization.quantize_dynamic(craft_net, dtype=torch.qint8, inplace=True)

    craft_net.to(device)
    craft_net.eval()
    return craft_net


# Preprocess the input image
def detect_text(image_path, model_path, device='cpu', text_threshold=0.7, link_threshold=0.4, low_text=0.4,
                canvas_size=2560, mag_ratio=1.5):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or cannot be opened.")

    # Convert image to RGB
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image while maintaining the aspect ratio
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio(img_rgb, canvas_size, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)

    craft_net = load_craft_model(model_path, device)

    with torch.no_grad():
        y, feature = craft_net(x)

    # Extract the text and link scores
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    # Get bounding boxes for detected text areas
    boxes, polys, _ = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly=False)

    # Adjust coordinates to the original image size
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)

    return boxes


# Helper function to draw boxes on the image
def draw_boxes(image_path, boxes):
    image = cv2.imread(image_path)
    for box in boxes:
        box = np.int32(box)
        cv2.polylines(image, [box.reshape((-1, 1, 2))], isClosed=True, color=(0, 255, 0), thickness=2)
    return image


def blur_text_regions(image_path, boxes, ksize=(25, 25)):
    image = cv2.imread(image_path)

    for box in boxes:
        box = np.int32(box).reshape((-1, 2))

        x_min, y_min = np.min(box[:, 0]), np.min(box[:, 1])
        x_max, y_max = np.max(box[:, 0]), np.max(box[:, 1])

        roi = image[y_min:y_max, x_min:x_max]

        blurred_roi = cv2.GaussianBlur(roi, ksize, 0)

        image[y_min:y_max, x_min:x_max] = blurred_roi
    return image


if __name__ == "__main__":
    image_path = 'D:\\Python_Project\\Masaike\\1.png'
    model_path = 'D:\\Python_Project\\Masaike\\utils\\craft_mlt_25k.pth'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    detected_boxes = detect_text(image_path, model_path, device=device)

    result_image = blur_text_regions(image_path, detected_boxes)

    output_path = 'output_image_with_blur.jpg'
    cv2.imwrite(output_path, result_image)
    print(output_path)
