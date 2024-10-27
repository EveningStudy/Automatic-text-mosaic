import torch
import numpy as np
from flask import Flask, request, render_template, send_from_directory, redirect, url_for, jsonify
import cv2
import os
import mosaicType

from utils.getTextbox import detect_text, draw_boxes

app = Flask(__name__)

# Define folders for image uploads and processed images
UPLOAD_FOLDER = './uploads'
PROCESSED_FOLDER = './processed'

# Ensure upload and processed directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Configure Flask application with folder paths
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Path to the pre-trained model for text detection
model_path = 'D:\\Python_Project\\Masaike\\utils\\craft_mlt_25k.pth'


# Main route for uploading images
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Check if file is in the POST request
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"

        if file:
            # Save uploaded file to the UPLOAD_FOLDER
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Set device based on CUDA availability
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            # Detect text in the image using the pre-trained model
            detected_boxes = detect_text(file_path, model_path, device=device)

            # Get mosaic type and color (default: GaussianBlur and black)
            mosaic_type = request.form.get('mosaic_type', 'GaussianBlur')
            color_hex = request.form.get('color', '#000000')
            color_rgb = tuple(int(color_hex[i:i + 2], 16) for i in (1, 3, 5))  # Convert color from hex to RGB

            # Save initial processed image and detected text boxes
            image = cv2.imread(file_path)
            processed_path = os.path.join(app.config['PROCESSED_FOLDER'], 'boxed_' + filename)
            cv2.imwrite(processed_path, image)
            box_data_path = os.path.join(app.config['PROCESSED_FOLDER'], 'boxes_' + filename + '.npy')
            np.save(box_data_path, detected_boxes)

            # Redirect to appropriate processing route based on selected action
            action = request.form.get('action')
            if action == 'automatic':
                return redirect(
                    url_for('automatic_mosaic', filename=filename, mosaic_type=mosaic_type, color=color_hex))
            elif action == 'manual':
                return redirect(url_for('select_text_region', filename='boxed_' + filename, mosaic_type=mosaic_type))

    return render_template('index.html')


# Route to select regions for manual text mosaic
@app.route('/select/<filename>', methods=['GET'])
def select_text_region(filename):
    mosaic_type = request.args.get('mosaic_type', 'GaussianBlur')
    return render_template('select.html', filename=filename, mosaic_type=mosaic_type)


# Apply mosaic effect to selected text regions manually
@app.route('/mosaic', methods=['POST'])
def mosaic_text_region():
    # Get coordinates and other necessary data from JSON request
    x = int(request.json['x'])
    y = int(request.json['y'])
    filename = request.json['filename']
    mosaic_type = request.json.get('mosaic_type', 'GaussianBlur')

    # Load detected boxes and the processed image
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    box_data_path = os.path.join(app.config['PROCESSED_FOLDER'], 'boxes_' + filename.replace('boxed_', '') + '.npy')
    detected_boxes = np.load(box_data_path, allow_pickle=True)
    image = cv2.imread(processed_path)

    # Calculate scaling factors based on original and displayed dimensions
    original_height, original_width = image.shape[:2]
    scaled_width = request.json['displayed_width']
    scaled_height = request.json['displayed_height']
    scale_x = original_width / scaled_width
    scale_y = original_height / scaled_height

    adjusted_x = int(x * scale_x)
    adjusted_y = int(y * scale_y)

    # Apply mosaic to the click box
    for box in detected_boxes:
        box = np.int32(box).reshape((-1, 2))
        x_min, y_min = np.min(box[:, 0]), np.min(box[:, 1])
        x_max, y_max = np.max(box[:, 0]), np.max(box[:, 1])
        if x_min <= adjusted_x <= x_max and y_min <= adjusted_y <= y_max:
            image = getattr(mosaicType, mosaic_type)(processed_path, [box])
            cv2.imwrite(processed_path, image)
            break
    return jsonify({'success': True})


# Route to save the manually processed image
@app.route('/save_manual_image/<filename>', methods=['GET'])
def save_manual_image(filename):
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=True)

# Add a new route to handle the Discord button action
@app.route('/discord_manual_action/<filename>', methods=['GET'])
def discord_manual_action(filename):

    return jsonify({'success': True})



# Route for automatic mosaic processing
@app.route('/automatic/<filename>', methods=['GET', 'POST'])
def automatic_mosaic(filename):
    mosaic_type = request.args.get('mosaic_type', 'GaussianBlur')
    color_hex = request.args.get('color', '#000000')
    color_rgb = tuple(int(color_hex[i:i + 2], 16) for i in (1, 3, 5))

    # Load the image and detected boxes for processing
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    detected_boxes = np.load(os.path.join(app.config['PROCESSED_FOLDER'], 'boxes_' + filename + '.npy'),
                             allow_pickle=True)

    # Apply the specified mosaic type
    if mosaic_type == 'ColorBlock':
        image = mosaicType.ColorBlock(file_path, detected_boxes, color=color_rgb)
    else:
        image = getattr(mosaicType, mosaic_type)(file_path, detected_boxes)

    # Save the processed image
    processed_image_path = os.path.join(app.config['PROCESSED_FOLDER'], 'processed_' + filename)
    cv2.imwrite(processed_image_path, image)
    return redirect(url_for('display_image', filename='processed_' + filename))


# Display and manage the final processed image
@app.route('/display/<filename>', methods=['GET', 'POST'])
def display_image(filename):
    if request.method == 'POST':
        action = request.form.get('action')
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)

        # Save or discard processed image based on action
        if action == 'save':
            return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=True)
        elif action == 'discard':
            os.remove(processed_path)
            # return "Image discarded!"
            return redirect(url_for('upload_image'))
        return redirect(url_for('upload_image'))

    return render_template('display.html', filename=filename)


# Routes for serving uploaded and processed files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=False , host='127.0.0.1', port=5050)


