import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Loading video
cap = cv2.VideoCapture('shape_test.mp4')

# Define font
font = cv2.FONT_HERSHEY_SIMPLEX

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
video_out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (416, 416))
classes = ["circle", "asterisk"]

while(cap.isOpened()):
    ret, img = cap.read()
    if not ret:
        break

    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
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
            if confidence > 0.2:  # Lowered confidence threshold
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2)
            cv2.putText(img, label, (x, y + 30), font, 2, (0,255,0), 3)

    # Write the frame into the file 'output.mp4'
    video_out.write(img)

    # cv2.imshow("Image", img)  # Commented out this line
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
video_out.release()  # Save the output
cv2.destroyAllWindows()