# Human Bone Fracture Identification

A deep learning-based web application for automated detection and classification of bone fractures in medical X-ray and MRI images using YOLOv11 object detection.

## 🎯 Features

- **Multi-class Fracture Detection**: Identifies 10 different types of bone fractures including:
  - Comminuted
  - Greenstick
  - Healthy (no fracture)
  - Linear
  - Oblique Displaced
  - Oblique
  - Segmental
  - Spiral
  - Transverse Displaced
  - Transverse

- **Web-based Interface**: User-friendly interface with drag-and-drop image upload
- **Real-time Detection**: Instant fracture detection with visual annotations
- **Confidence Scoring**: Displays confidence scores for each detection
- **REST API**: Easy-to-use API endpoint for integration with other systems
- **Visual Feedback**: Annotated images with bounding boxes and labels

## 🛠️ Technology Stack

- **Deep Learning Framework**: [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- **Backend**: Flask (Python web framework)
- **Frontend**: HTML, CSS, JavaScript
- **Computer Vision**: OpenCV, Pillow
- **Experiment Tracking**: Comet ML
- **Dataset**: Human Bone Fractures Multi-modal Image Dataset (HBFMID) - 1,539 annotated images

## 📋 Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- Required Python packages:
  - ultralytics
  - flask
  - flask-cors
  - opencv-python
  - pillow
  - numpy
  - comet-ml

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/bobbymcclosky/HumanBoneFractureIdentification.git
   cd HumanBoneFractureIdentification
   ```

2. **Install dependencies**
   ```bash
   pip install ultralytics flask flask-cors opencv-python pillow numpy comet-ml
   ```

3. **Download the pre-trained model**
   
   The trained model weights (`bestRun.pt`) should be placed in the root directory. If not present, you'll need to train the model first (see Training section).

## 💻 Usage

### Running the Web Application

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Access the web interface**
   
   Open your browser and navigate to: `http://localhost:5000`

3. **Upload and analyze images**
   - Click the upload area or drag and drop an X-ray/MRI image
   - Supported formats: PNG, JPG, JPEG, DICOM
   - Click "Analyze Image" to detect fractures
   - View results with annotated bounding boxes and confidence scores

### Using the API

**Endpoint**: `POST /predict`

**Request**:
```bash
curl -X POST -F "file=@path/to/xray.jpg" http://localhost:5000/predict
```

**Response**:
```json
{
  "detections": [
    {
      "class": "Transverse",
      "confidence": 0.87,
      "bbox": {
        "x1": 120,
        "y1": 80,
        "x2": 340,
        "y2": 290
      }
    }
  ],
  "num_detections": 1,
  "annotated_image": "data:image/png;base64,iVBORw0KG..."
}
```

**Health Check Endpoint**: `GET /health`
```bash
curl http://localhost:5000/health
```

## 🎓 Training the Model

### Full Training

Train on the complete dataset with data augmentation and transfer learning:

```bash
python train.py
```

**Training Configuration**:
- Epochs: 100
- Image size: 640x640
- Batch size: 8
- Initial learning rate: 0.001
- Frozen layers: 10 (transfer learning)
- Patience: 15 (early stopping)
- Dropout: 0.5
- Data augmentation: Mosaic (50%), MixUp (10%)

### Overfitting Check

Before full training, verify the model can learn by overfitting a small dataset:

1. **Create small dataset**
   ```bash
   python createSmallDS.py
   ```

2. **Run overfitting check**
   ```bash
   python trainOverfit.py
   ```

This trains on 50 images without regularization. Success is indicated by training loss approaching zero.

## 📊 Dataset Information

- **Source**: Roboflow - Bone Fracture Detection Dataset v2
- **Total Images**: 1,539 annotated medical images
- **Format**: YOLOv8/YOLOv11 (YOLO format annotations)
- **Classes**: 10 fracture types
- **Splits**: Train, Validation, Test
- **Pre-processing**:
  - Auto-orientation (EXIF stripping)
  - Resize to 640x640 (stretch)
- **Augmentation** (3x per source image):
  - Horizontal flip (50% probability)
  - Vertical flip (50% probability)
  - Random rotation (-5° to +5°)
  - Random shear (-2° to +2°)
  - Random brightness adjustment (-10% to +10%)

## 🏗️ Project Structure

```
HumanBoneFractureIdentification/
├── app.py                          # Flask web application server
├── train.py                        # Main training script
├── trainOverfit.py                 # Overfitting validation script
├── createSmallDS.py                # Small dataset creation utility
├── index.html                      # Web interface
├── bestRun.pt                      # Trained model weights
├── yolo11n.pt                      # Pre-trained YOLOv11 nano base model
├── runs/                           # Training run outputs
│   └── detect/                     # Detection model runs
│       ├── fracture_best_model*/   # Training checkpoints
│       └── overfit_check/          # Overfitting test results
└── Human Bone Fractures Multi-modal Image Dataset (HBFMID)/
    ├── Bone Fractures Detection/
    │   ├── data.yaml               # Dataset configuration
    │   ├── train/                  # Training images and labels
    │   ├── valid/                  # Validation images and labels
    │   └── test/                   # Test images and labels
    └── Bone_Fractures_Overfit_Check/
        └── data_small.yaml         # Small dataset configuration
```

## 🔬 Model Architecture

- **Base Model**: YOLOv11 Nano (yolo11n)
- **Transfer Learning**: First 10 layers frozen
- **Input Size**: 640x640 pixels
- **Output**: Bounding boxes with class predictions and confidence scores
- **Optimization**: Adam optimizer with cosine annealing learning rate schedule
- **Loss Functions**: 
  - Bounding box regression loss
  - Classification loss
  - Objectness loss

## 📈 Model Performance

The model is trained with early stopping based on validation performance. Key metrics tracked:
- mAP (mean Average Precision)
- Precision
- Recall
- Training/Validation loss

Training progress and metrics are tracked using Comet ML for visualization and analysis.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 👤 Author

**Bobby McClosky**

## 📝 License

This project uses the Human Bone Fractures Multi-modal Image Dataset (HBFMID) which is licensed as Private on Roboflow. Please respect the dataset license terms.

## 🙏 Acknowledgments

- Dataset provided by [Roboflow](https://roboflow.com/) - IUBAT workspace
- YOLOv11 by [Ultralytics](https://github.com/ultralytics/ultralytics)
- Training metrics tracking by [Comet ML](https://www.comet.ml/)

## ⚠️ Disclaimer

This tool is designed for educational and research purposes. It should not be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare professionals for medical decisions.
