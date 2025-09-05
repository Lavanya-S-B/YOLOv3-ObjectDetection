import cv2
import numpy as np
import imutils
import pyttsx3

# --- Text-to-speech function (optional) ---
def tts(text):
    engine = pyttsx3.init()
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate)
    engine.say(text)
    engine.runAndWait()

# --- Paths to YOLO model files ---
cfg_path = "yolov3.cfg"
weights_path = "yolov3.weights"
names_path = "coco.names"

# --- Load YOLO model ---
net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)

# Load class names
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# --- Accuracy settings ---
confidence_threshold = 0.6   # higher = fewer false positives
nms_threshold = 0.3          # lower = stricter suppression

# --- Start webcam ---
cap = cv2.VideoCapture(0)  # use 0 for default webcam else 1

print("Press 'q' or 'ESC' to quit.")

while True:
    ret, img = cap.read()
    if not ret:
        break

    img = imutils.resize(img, width=700)  # resize for display
    height, width, channels = img.shape

    # Create blob with higher resolution (608×608 for accuracy)
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (608, 608), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Max Suppression (NMS)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    font = cv2.FONT_HERSHEY_PLAIN
    object_count = 0

    for i in range(len(boxes)):
        if i in indexes:
            object_count += 1
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]

            # Draw box and label
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10), font, 2, color, 2)

            print(f"Detected: {label}")
            # Uncomment if you want speech
            # tts(label)

    # Show detection results
    cv2.imshow("YOLOv3 Accurate Detection", img)

    # Exit keys (q or ESC)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break

# --- Clean up ---
cap.release()
cv2.destroyAllWindows()
print("Program stopped safely.")
