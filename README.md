# Chest X-Ray TB Detection & Analysis System

An AI-powered system for detecting Tuberculosis (TB) from Chest X-Ray images, featuring a **Flutter-based mobile/web application**, a **Python Flask backend**, and **optimized AI models**.

![Project Banner](docs/Software_Copyright_Forms_html_e10ca323.png) 
*(Replace with actual screenshot if available)*

## 🚀 Features

-   **Cross-Platform App**: Flutter app for Android, iOS, and Web.
-   **AI Analysis**: DenseNet121-based model for accurate TB classification (Normal vs. TB).
-   **Radiology Reports**: automatic generation of detailed PDF radiology reports.
-   **Offline Inference**: TFLite support for on-device analysis (mobile).
-   **Web API**: Flask-based API for web inference and heavy-lifting.
-   **Comprehensive Evaluation**: Scripts for calculating Accuracy, Sensitivity, Specificity, and Confusion Matrix.

## 📂 Project Structure

```
.
├── app/              # Flutter Application (Dart)
├── backend/          # Python Flask API & Model Logic
├── data/             # Dataset and Accuracy Results
├── docs/             # Documentation, Research Papers, & Copyright Forms
├── models/           # Trained AI Models (.pt, .onnx, .tflite)
├── scripts/          # Utility Scripts (Training, Evaluation, Conversion)
└── README.md         # Project Documentation
```

## 🛠️ Installation & Setup

### 1. Prerequisites
-   **Git** installed.
-   **Python 3.8+** installed.
-   **Flutter SDK** installed.

### 2. Backend Setup (Python)

Navigate to the `backend` directory and set up the environment:

```bash
cd backend
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (ensure you are in the root or backend dir where requirements exists)
pip install -r requirements.txt
```

**Note**: You may need to install `torch` and `torchvision` manually matching your CUDA version if you have a GPU.

To run the API server:
```bash
python api.py
```
The server will start at `http://0.0.0.0:5000`.

### 3. App Setup (Flutter)

Navigate to the `app` directory:

```bash
cd app
flutter pub get
```

To run the app:
```bash
# For Web (requires backend running for API mode)
flutter run -d chrome

# For Android (supports offline TFLite)
flutter run -d android
```

## 🧠 AI Models

The project uses a fine-tuned **DenseNet121** model. Key files in `models/`:
-   `tb_model_best.pt`: PyTorch model (Best Weights).
-   `tb_model_best.onnx`: ONNX format.
-   `tb_model_best.tflite`: TFLite format for mobile.

## 📊 Evaluation

To verify model accuracy and generate metrics:

```bash
cd scripts
python verify_accuracy.py
python evaluate_model.py
```
Results (Confusion Matrix, Metrics) will be saved in `docs/`.

## 📄 Documentation

-   **Research Paper**: See `docs/research_paper_polished.pdf`.
-   **Copyright**: See `docs/Software_Copyright_Forms.pdf`.

## 🤝 Credits

-   **Frameworks**: PyTorch, Flutter, Flask.
-   **Model Architecture**: DenseNet121.
