"""
Human Bone Fracture Identification - Web Application Server

This Flask application provides a web-based interface for detecting bone fractures
in medical images using a trained YOLOv11 object detection model. The application
accepts X-ray or MRI images, runs inference, and returns annotated results with
bounding boxes and confidence scores.

Features:
- REST API endpoint for fracture detection
- Real-time image processing and annotation
- Support for multiple image formats (JPG, PNG, DICOM)
- Visual feedback with bounding boxes and labels
- Confidence scoring for each detection

Technology Stack:
- Flask: Web framework and routing
- YOLOv11: Object detection model (Ultralytics)
- OpenCV: Image processing and annotation
- Pillow: Image format handling

Author: Bobby McClosky
API Version: 1.0
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io
import cv2
import numpy as np
import base64

# ============================================================================
# Application Configuration
# ============================================================================
# Initialize Flask application with static file serving
app = Flask(__name__, static_folder='.', static_url_path='')

# Enable Cross-Origin Resource Sharing (CORS) for API access
CORS(app)

# ============================================================================
# Model Loading
# ============================================================================
# Load the trained YOLOv11 model for bone fracture detection
# Note: Ensure 'bestRun.pt' exists in the application directory
model = YOLO('bestRun.pt')  # Pre-trained model weights

# ============================================================================
# Web Routes
# ============================================================================

@app.route('/')
def serve_index():
    """
    Serve the main HTML interface for the application.
    
    Returns:
        HTML file: The index.html file from the static directory
    """
    return send_from_directory('.', 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Process uploaded medical images and detect bone fractures.
    
    This endpoint accepts an image file via POST request, runs YOLOv11 inference
    to detect fractures, and returns the detection results along with an
    annotated version of the image.
    
    Request:
        - Method: POST
        - Content-Type: multipart/form-data
        - Body: 'file' parameter containing the image
        
    Returns:
        JSON response containing:
        - detections: List of detected fractures with bounding boxes and confidence
        - num_detections: Total number of fractures detected
        - annotated_image: Base64-encoded image with drawn bounding boxes
        
    Error Responses:
        - 400: No file provided or invalid image format
        - 500: Internal server error during processing
        
    Example Response:
        {
            "detections": [
                {
                    "class": "fracture",
                    "confidence": 0.87,
                    "bbox": {"x1": 120, "y1": 80, "x2": 340, "y2": 290}
                }
            ],
            "num_detections": 1,
            "annotated_image": "data:image/png;base64,iVBORw0KG..."
        }
    """
    try:
        # ====================================================================
        # Input Validation
        # ====================================================================
        # Check if file is present in the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # ====================================================================
        # Image Processing
        # ====================================================================
        # Read and decode the uploaded image
        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        # Validate image decoding
        if img is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        # ====================================================================
        # Model Inference
        # ====================================================================
        # Run YOLOv11 detection with 50% confidence threshold
        results = model.predict(img, conf=0.5)
        
        # ====================================================================
        # Parse Detection Results
        # ====================================================================
        if len(results) > 0:
            result = results[0]
            
            # Extract detection information (bounding boxes, classes, confidence)
            detections = []
            if result.boxes is not None:
                for box in result.boxes:
                    # Get bounding box coordinates (x1, y1, x2, y2)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Get confidence score
                    conf = float(box.conf[0])
                    
                    # Get class ID and name
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    
                    # Store detection information
                    detections.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': {
                            'x1': x1,
                            'y1': y1,
                            'x2': x2,
                            'y2': y2
                        }
                    })
            
            # ================================================================
            # Image Annotation
            # ================================================================
            # Create a copy of the original image for annotation
            img_annotated = img.copy()
            
            # Draw bounding boxes and labels on the image
            for detection in detections:
                bbox = detection['bbox']
                x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
                conf = detection['confidence']
                label = detection['class']
                
                # Draw green rectangle around detected fracture
                cv2.rectangle(img_annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label with confidence score above the bounding box
                cv2.putText(img_annotated, f'{label} {conf:.2f}', (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # ================================================================
            # Encode Annotated Image
            # ================================================================
            # Convert annotated image to PNG format
            _, img_encoded = cv2.imencode('.png', img_annotated)
            
            # Encode as base64 string for web display
            img_base64 = base64.b64encode(img_encoded).decode()
            
            # Return detection results and annotated image
            return jsonify({
                'detections': detections,
                'num_detections': len(detections),
                'annotated_image': f'data:image/png;base64,{img_base64}'
            })
        
        # No detections found
        return jsonify({'detections': [], 'num_detections': 0, 'annotated_image': None})
        
    except Exception as e:
        # Handle any unexpected errors
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint for monitoring application status.
    
    Returns:
        JSON response: {"status": "ok"} with 200 status code
        
    Usage:
        Used by monitoring systems and load balancers to verify
        that the application is running and responsive.
    """
    return jsonify({'status': 'ok'})


# ============================================================================
# Application Entry Point
# ============================================================================
if __name__ == '__main__':
    """
    Start the Flask development server.
    
    Configuration:
    - Debug mode: Enabled (provides detailed error messages and auto-reload)
    - Port: 5000 (default Flask port)
    - Host: localhost (127.0.0.1)
    
    Note: For production deployment, use a production-grade WSGI server
    such as Gunicorn or uWSGI instead of the Flask development server.
    """
    app.run(debug=True, port=5000)
