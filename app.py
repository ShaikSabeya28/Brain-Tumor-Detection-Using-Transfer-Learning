from flask import Flask, request, jsonify, render_template, url_for
import numpy as np
from flask import send_from_directory
from PIL import Image
import tensorflow as tf
import os
import random
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='mri-images')
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

CNN = None  # global variable to hold the CNN model

def load_model():
    global CNN
    if CNN is not None:
        return
    script_dir = os.path.dirname(__file__)
    model_json_path = os.path.join(script_dir, 'models', 'CNN_structure.json')
    with open(model_json_path, 'r') as json_file:
        model_json = json_file.read()
    try:
        CNN = tf.keras.models.model_from_json(model_json)
        weights_path = os.path.join(script_dir, 'models', 'CNN_weights.pkl')
        with open(weights_path, 'rb') as weights_file:
            weights = pickle.load(weights_file)
            CNN.set_weights(weights)
        CNN.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001), 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'])
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")

def get_model_prediction(image_path):
    load_model()
    try:
        img = Image.open(image_path).resize((224, 224))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_array = np.expand_dims(np.array(img), axis=0)
        prediction = CNN.predict(img_array)
        predicted_index = np.argmax(prediction[0])
        class_labels = ['glioma', 'meningioma', 'no tumor', 'pituitary']
        return class_labels[predicted_index]
    except Exception as e:
        logger.error(f"Error in get_model_prediction: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Existing random image route
@app.route('/get-random-image', methods=['GET'])
def get_random_image():
    try:  
        class_dirs = ['glioma', 'meningioma', 'notumor', 'pituitary']
        selected_class = random.choice(class_dirs)
        image_dir = os.path.join('mri-images', selected_class)
        image_name = random.choice(os.listdir(image_dir))
        image_path = os.path.join(image_dir, image_name)
        predicted_label = get_model_prediction(image_path)
        web_accessible_image_path = url_for('static', filename=f'{selected_class}/{image_name}')
        return jsonify({
            'image_path': web_accessible_image_path,
            'actual_label': selected_class,
            'predicted_label': predicted_label
        })
    except Exception as e:
        logger.error(f"Error in get-random-image route: {e}")
        return jsonify({'error': 'An error occurred'}), 500

# New upload route
@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        predicted_label = get_model_prediction(file_path)
        # use web_accessible_image_path format for uploaded images
        web_accessible_image_path = url_for('uploaded_file', filename=file.filename)

        return jsonify({
            'image_path': web_accessible_image_path,
            'predicted_label': predicted_label
        })
    except Exception as e:
        logger.error(f"Error in upload-image route: {e}")
        return jsonify({'error': 'Prediction failed'}), 500

if __name__ == '__main__':
    app.run(debug=False)
