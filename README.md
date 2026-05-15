# RoboVision — Real-Time Object Detection

A camera-based vision system that identifies objects in real time using your webcam — 
built with TensorFlow.js and the COCO-SSD model, just like a robot vision system.

# Features
- Detects 80+ object classes (people, animals, electronics, vehicles, food, and more)
- Live bounding boxes with confidence scores
- Adjustable confidence threshold
- FPS counter and inference time display
- Detection log with timestamps
- No installation needed — runs entirely in the browser

# How to Run
1. Clone or download this repo
2. Open the folder in VS Code
3. Install the Live Server extension
4. Right-click `index.html` → Open with Live Server
5. Allow camera access and click START CAMERA

# Tech Stack
- TensorFlow.js
- COCO-SSD pre-trained model
- Vanilla HTML, CSS, JavaScript

# How It Works
Webcam → TensorFlow.js captures frames → COCO-SSD runs inference → 
Bounding boxes drawn on canvas overlay → UI updates in real time
