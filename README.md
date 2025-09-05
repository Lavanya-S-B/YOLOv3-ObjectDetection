YOLOv3 Object Detection

This project implements real-time object detection using the YOLOv3 (You Only Look Once) deep learning model with OpenCV and Python. 
YOLOv3 is a state-of-the-art, fast, and accurate object detection algorithm capable of detecting multiple objects in an image or video stream.

Project Structure

ObjectDetection

│── main.py                # Python script to run YOLOv3 detection

│── yolov3.cfg             # YOLOv3 model configuration file

│── yolov3.weights         # Pre-trained weights (not included in repo, see below)

│── yolov3.txt             # COCO class labels file

│── README.md              # Project documentation

Download Pre-trained YOLOv3 Weights

The pre-trained weights are too large to store in GitHub.
Download them manually from the official YOLO website:

👉 Download YOLOv3 Weights: https://pjreddie.com/media/files/yolov3.weights

Place the downloaded file in your project folder.

📊 Results

Detects 80+ object categories from the COCO dataset.

Provides bounding boxes, class labels, and confidence scores.

Works on both images and real-time video streams.
