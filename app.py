import torch
import numpy as np
from flask import Flask, request, render_template, send_from_directory, redirect, url_for, jsonify
import cv2
import os
import mosaicType

from utils.getTextbox import detect_text, draw_boxes

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
PROCESSED_FOLDER = './processed'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

model_path = 'D:\\Python_Project\\Masaike\\utils\\craft_mlt_25k.pth'


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"

        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            detected_boxes = detect_text(file_path, model_path, device=device)

            mosaic_type = request.form.get('mosaic_type', 'GaussianBlur')

            color_hex = request.form.get('color', '#000000')
            color_rgb = tuple(int(color_hex[i:i + 2], 16) for i in (1, 3, 5))  # 转换为RGB

            image_with_boxes = draw_boxes(file_path, detected_boxes)
            processed_path = os.path.join(app.config['PROCESSED_FOLDER'], 'boxed_' + filename)
            cv2.imwrite(processed_path, image_with_boxes)

            box_data_path = os.path.join(app.config['PROCESSED_FOLDER'], 'boxes_' + filename + '.npy')
            np.save(box_data_path, detected_boxes)

            action = request.form.get('action')
            if action == 'automatic':
                if mosaic_type == 'ColorBlock':
                    image = mosaicType.ColorBlock(file_path, detected_boxes, color=color_rgb)
                else:
                    image = getattr(mosaicType, mosaic_type)(file_path, detected_boxes)

                processed_image_path = os.path.join(app.config['PROCESSED_FOLDER'], 'processed_' + filename)
                cv2.imwrite(processed_image_path, image)
                return redirect(url_for('display_image', filename='processed_' + filename))
            elif action == 'manual':
                return redirect(url_for('select_text_region', filename='boxed_' + filename, mosaic_type=mosaic_type))

    return render_template('index.html')


@app.route('/select/<filename>', methods=['GET'])
def select_text_region(filename):
    mosaic_type = request.args.get('mosaic_type', 'GaussianBlur')
    return render_template('select.html', filename=filename, mosaic_type=mosaic_type)


@app.route('/mosaic', methods=['POST'])
def mosaic_text_region():
    # 获取用户点击的坐标
    x = int(request.json['x'])
    y = int(request.json['y'])
    filename = request.json['filename']
    mosaic_type = request.json.get('mosaic_type', 'GaussianBlur')

    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)

    box_data_path = os.path.join(app.config['PROCESSED_FOLDER'], 'boxes_' + filename.replace('boxed_', '') + '.npy')
    detected_boxes = np.load(box_data_path, allow_pickle=True)

    for box in detected_boxes:
        box = np.int0(box).reshape((-1, 2))
        x_min, y_min = np.min(box[:, 0]), np.min(box[:, 1])
        x_max, y_max = np.max(box[:, 0]), np.max(box[:, 1])

        if x_min <= x <= x_max and (y_min - 32) <= y <= (y_max - 32):
            image = getattr(mosaicType, mosaic_type)(processed_path, [box])
            cv2.imwrite(processed_path, image)
            break

    return jsonify({'success': True})


@app.route('/display/<filename>', methods=['GET', 'POST'])
def display_image(filename):
    if request.method == 'POST':
        action = request.form.get('action')
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)

        if action == 'save':
            return "Image saved successfully! You can download it from the processed folder."
        elif action == 'discard':
            os.remove(processed_path)
            return "Image discarded!"

        return redirect(url_for('upload_image'))

    return render_template('display.html', filename=filename)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
