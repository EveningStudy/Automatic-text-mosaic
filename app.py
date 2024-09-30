from flask import Flask, request, render_template, send_from_directory, redirect, url_for
import cv2
import os
import mosaicType

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
PROCESSED_FOLDER = './processed'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

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

            image = cv2.imread(file_path)
            padding = 5
            lang = ['ch_sim', 'en']
            image = mosaicType.common(image, padding, lang)

            processed_path = os.path.join(app.config['PROCESSED_FOLDER'], 'processed_' + filename)
            cv2.imwrite(processed_path, image)

            return redirect(url_for('display_image', filename='processed_' + filename))

    return render_template('index.html')


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
