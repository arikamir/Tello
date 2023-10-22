import cv2
import numpy as np

# Load label names
with open("src/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load YOLO model
net = cv2.dnn.readNet("src/yolov3.weights", "config/yolov3.cfg")
layer_names = net.getLayerNames()
print(net.getUnconnectedOutLayers())
output_layers = [layer_names[i- 1] for i in net.getUnconnectedOutLayers()]

# # Initialize the webcam
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()

# # Capture a single frame
# ret, frame = cap.read()
# if not ret:
#     print("Error reading frame")

# # Release the webcam
# cap.release()

#Load an image
frame = cv2.imread("src/object_detection_images/istockphoto-1293762741-612x612.jpeg")
height, width, channels = frame.shape

# Detect objects
blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Information to show on the object (class id, confidence)
class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.45:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

            color = (0, 255, 0)  # Green
            label = str(classes[class_id])
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

cv2.imshow("Frame", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
