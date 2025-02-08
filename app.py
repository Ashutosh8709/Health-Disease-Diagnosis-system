from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# Model configuration
MODEL_CONFIG = {
    'osteoarthritis': {
        'input_shape': (224, 224, 3),
        'color_mode': 'rgb',
        'classes': ['Normal', 'Osteopenia', 'Osteoporosis']
    },
    'retinopathy': {
        'input_shape': (224, 224, 3),
        'color_mode': 'rgb',
        'classes': ['No DR', 'DR']
    },
    'brain_tumor': {
        'input_shape': (224, 224, 3),
        'color_mode': 'rgb',
        'classes': ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    },
    'chest_xray': {
        'input_shape': (224, 224, 3),
        'color_mode': 'rgb',
        'classes': ['COVID-19', 'Normal', 'Pneumonia', 'Tuberculosis']
    }
}

# Load models
models = {}
for model_name, config in MODEL_CONFIG.items():
    try:
        model_path = os.path.join('models', f'{model_name}.h5')
        models[model_name] = tf.keras.models.load_model(model_path)
        print(f"✅ Loaded {model_name} model successfully")
    except Exception as e:
        print(f"❌ Error loading {model_name} model: {str(e)}")
        raise

def preprocess_image(image, model_type):
    config = MODEL_CONFIG[model_type]
    image = image.resize(config['input_shape'][:2])
    image = image.convert(config['color_mode'].upper())
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    if len(img_array.shape) == 2:
        img_array = np.expand_dims(img_array, axis=-1)
        
    return np.expand_dims(img_array, axis=0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload/<model_type>')
def upload(model_type):
    return render_template('upload.html',
                         model_type=model_type,
                         model_config=MODEL_CONFIG[model_type],
                         model_name=model_type.replace('_', ' ').title())

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model_type = request.form['model_type']
        config = MODEL_CONFIG[model_type]
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
            
        image = request.files['image'].read()
        img = Image.open(io.BytesIO(image))
        processed_img = preprocess_image(img, model_type)
        
        # Validate input shape
        if processed_img.shape[1:] != config['input_shape']:
            raise ValueError(f"Invalid input shape. Expected {config['input_shape']}, got {processed_img.shape[1:]}")
        
        prediction = models[model_type].predict(processed_img)
        class_idx = np.argmax(prediction)
        
        return jsonify({
            'model': model_type,
            'prediction': config['classes'][class_idx],
            'confidence': float(np.max(prediction)),
            'all_predictions': prediction[0].tolist()
        })
        
    except KeyError:
        return jsonify({'error': 'Invalid model type'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
