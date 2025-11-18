from flask import Flask, render_template, request
from keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
model = load_model('crop_disease_model.h5')  # <-- Your trained model file

# Class names (same order as training)
class_names = [
    "Pepper Bell - Bacterial Spot",
    "Pepper Bell - Healthy",
    "Potato - Early Blight",
    "Potato - Healthy",
    "Potato - Late Blight",
    "Tomato - Target Spot",
    "Tomato - Mosaic Virus",
    "Tomato - Yellow Leaf Curl Virus",
    "Tomato - Bacterial Spot",
    "Tomato - Early Blight",
    "Tomato - Healthy",
    "Tomato - Late Blight",
    "Tomato - Leaf Mold",
    "Tomato - Septoria Leaf Spot",
    "Tomato - Spider Mites (Two Spotted Spider Mite)"
]

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Image preprocessing
def preprocess_image(file_path):
    img = Image.open(file_path)
    img = img.convert('RGB')
    img = img.resize((128, 128))  # same size as training
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 128, 128, 3)
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Preprocess and predict
    img_array = preprocess_image(file_path)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = class_names[predicted_class]
    confidence = round(np.max(prediction) * 100, 2)

    return render_template('result.html',
                           prediction=predicted_label,
                           confidence=confidence,
                           img_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)
