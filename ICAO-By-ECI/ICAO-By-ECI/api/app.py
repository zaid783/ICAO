from flask import Flask, render_template, request, flash, send_file
from werkzeug.utils import secure_filename
import os
import zipfile
from io import BytesIO
import cv2
import numpy as np
import os
from PIL import Image
from rembg import remove
import cv2.data

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.secret_key = 'super secret key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER



def resize_and_crop_to_icao(img, target_size=(827, 1063)):
    target_ratio = target_size[0] / target_size[1]
    img_ratio = img.width / img.height

    if img_ratio > target_ratio:
        new_height = target_size[1]
        new_width = int(new_height * img_ratio)
    else:
        new_width = target_size[0]
        new_height = int(new_width / img_ratio)

    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    new_img = Image.new('RGB', target_size, (255, 255, 255))
    offset = ((target_size[0] - new_width) // 2, (target_size[1] - new_height) // 2)
    new_img.paste(img, offset)
    return new_img

def remove_background(img):
    img_array = np.array(img)
    img_no_bg_array = remove(img_array)
    img_no_bg_pil = Image.fromarray(img_no_bg_array, 'RGBA')
    white_background = Image.new('RGBA', img_no_bg_pil.size, (255, 255, 255, 255))
    img_with_white_bg = Image.alpha_composite(white_background, img_no_bg_pil)
    img_with_white_bg = img_with_white_bg.convert("RGB")
    return img_with_white_bg

def detect_and_crop_face(img):
    img_cv = np.array(img)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print("No face detected.")
        return img

    (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
    padding_w = int(0.1 * w)
    padding_h = int(0.1 * h)
    x = max(x - padding_w, 0)
    y = max(y - padding_h, 0)
    w = min(w + 2 * padding_w, img_cv.shape[1] - x)
    h = min(h + 2 * padding_h, img_cv.shape[0] - y)
    face_img = img_cv[y:y + h, x:x + w]
    face_img_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    return face_img_pil

def process_images_for_icao(input_folder, output_folder, target_size=(827, 1063), dpi=600):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.png')
            try:
                with Image.open(image_path) as img:
                    img = img.convert("RGB")
                    img_no_bg = remove_background(img)
                    processed_img = detect_and_crop_face(img_no_bg)
                    face_img = resize_and_crop_to_icao(processed_img, target_size)

                    if face_img is not None:
                        face_img.save(output_path, format='PNG', dpi=(dpi, dpi))
                        print(f"Processed: {filename}")
                    else:
                        print(f"Face not detected in: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_zip(file_paths):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        for file in file_paths:
            zip_file.write(file, os.path.basename(file))
    zip_buffer.seek(0)
    return zip_buffer


def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/edit", methods=["GET", "POST"])
def edit():
    if request.method == "POST":
        if 'files' not in request.files or len(request.files.getlist('files')) == 0:
            flash('No files selected for uploading')
            return render_template("index.html")

        files = request.files.getlist('files')

        # Clear old files in UPLOAD_FOLDER and PROCESSED_FOLDER
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        if not os.path.exists(app.config['PROCESSED_FOLDER']):
            os.makedirs(app.config['PROCESSED_FOLDER'])
        clear_folder(app.config['UPLOAD_FOLDER'])
        clear_folder(app.config['PROCESSED_FOLDER'])

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

        # Process images using backend_file.py functions
        process_images_for_icao(app.config['UPLOAD_FOLDER'], app.config['PROCESSED_FOLDER'])

        processed_files = [os.path.join(app.config['PROCESSED_FOLDER'], file) for file in
                           os.listdir(app.config['PROCESSED_FOLDER']) if allowed_file(file)]

        if processed_files:
            zip_buffer = create_zip(processed_files)
            return send_file(zip_buffer, as_attachment=True, download_name='processed_images.zip',
                             mimetype='application/zip')

        flash("No valid files processed.")
        return render_template("index.html")

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True,port=5000)