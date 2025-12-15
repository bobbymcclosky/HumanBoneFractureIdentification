from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io
import cv2
import numpy as np
import base64

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Load your trained model
model = YOLO('bestRun.pt')  # Make sure best.pt is in the same directory

@app.route('/')
def serve_index():
    """Serve the main HTML file"""
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Read image
        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        # Run inference
        results = model.predict(img, conf=0.5)
        
        # Parse detection results
        if len(results) > 0:
            result = results[0]
            
            # Get detections
            detections = []
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    
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
            
            # Draw boxes on image for visualization
            img_annotated = img.copy()
            for detection in detections:
                bbox = detection['bbox']
                x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
                conf = detection['confidence']
                label = detection['class']
                
                # Draw rectangle
                cv2.rectangle(img_annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Draw label
                cv2.putText(img_annotated, f'{label} {conf:.2f}', (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Convert annotated image to base64 for display
            _, img_encoded = cv2.imencode('.png', img_annotated)
            img_base64 = base64.b64encode(img_encoded).decode()
            
            return jsonify({
                'detections': detections,
                'num_detections': len(detections),
                'annotated_image': f'data:image/png;base64,{img_base64}'
            })
        
        return jsonify({'detections': [], 'num_detections': 0, 'annotated_image': None})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
