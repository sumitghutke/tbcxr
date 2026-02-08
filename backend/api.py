import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import DenseNet121  # Importing the model definition

app = Flask(__name__)
CORS(app) # Enable CORS for all routes (important for Web)

# --- Model Loading ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/tb_model_best.pt")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    print(f"Loading model from {MODEL_PATH} to {DEVICE}...")
    model = DenseNet121()
    # The saved state_dict might need mapping if saved on GPU and loading on CPU
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.model.load_state_dict(state_dict) 
    model.to(DEVICE)
    model.eval()
    return model

try:
    model = load_model()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- Preprocessing ---
def transform_image(image_bytes):
    # Standard transforms matching training
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_bytes).convert("RGB")
    return transform(image).unsqueeze(0).to(DEVICE)

# --- Routes ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        input_tensor = transform_image(file)
        
        with torch.no_grad():
            output = model(input_tensor) # Logits
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_class_idx = torch.max(probabilities, 1)
            
        label = "Tuberculosis" if predicted_class_idx.item() == 1 else "Normal"
        conf_val = confidence.item()
        
        return jsonify({
            'label': label,
            'confidence': conf_val
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def health_check():
    return "Chest X-Ray API is running."

if __name__ == '__main__':
    # Run on 0.0.0.0 to be accessible easily
    app.run(host='0.0.0.0', port=5000)
