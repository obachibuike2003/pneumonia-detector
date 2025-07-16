import os
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
# --- CORRECTED IMPORT ---
from keras.preprocessing.image import load_img, img_to_array 
# ------------------------

# Initialize Flask app
app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = 'uploads' # Folder to temporarily store uploaded uploaded images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- Load the Trained Model ---
MODEL_PATH = 'pneumonia_model.h5'
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Please ensure '{MODEL_PATH}' exists in the same directory as app.py and is not corrupted.")
    exit() # Exit if model cannot be loaded

# --- Model Specifics (from your pneumonia_classifier.py) ---
TARGET_SIZE = (150, 150) # The size your model expects images to be

# --- Helper Function to Check Allowed File Types ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main page for image upload."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles image upload, runs prediction, and returns results."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        # Secure filename and save temporarily
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        try:
            # Preprocess the image
            img = load_img(filename, target_size=TARGET_SIZE)
            img_array = img_to_array(img)
            # Add a batch dimension. Your model has Rescaling(1./255) layer built-in.
            img_array = np.expand_dims(img_array, axis=0)

            # Make prediction
            prediction = model.predict(img_array)
            # The model outputs a single value between 0 and 1 (sigmoid activation)
            confidence_raw = float(prediction[0][0]) # This is the probability of Pneumonia

            # Determine diagnosis and display confidence appropriately
            if confidence_raw > 0.5: # Threshold for classification
                diagnosis = "Pneumonia"
                confidence_display = confidence_raw * 100
            else:
                diagnosis = "Normal"
                confidence_display = (1 - confidence_raw) * 100 # Confidence for Normal

            result = {
                'diagnosis': diagnosis,
                'confidence': f"{confidence_display:.2f}%"
            }
            return jsonify(result)

        except Exception as e:
            print(f"Prediction error: {e}")
            return jsonify({'error': f'Error processing image for prediction: {str(e)}'}), 500
        finally:
            # Clean up: remove the temporary file
            if os.path.exists(filename):
                os.remove(filename)
    else:
        return jsonify({'error': 'Allowed image types are PNG, JPG, JPEG'}), 400

# --- Run the Flask App ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)