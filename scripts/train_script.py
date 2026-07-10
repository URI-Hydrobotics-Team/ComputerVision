"""
1.Install ultralytics to access YOLO models
"""

"""2.Import YOLO to access models"""

from ultralytics import YOLO
import cv2 as cv

# CHANGE TO THE DESIRED epochs
epochs = 300

# CHANGE TO THE actual best.pt file location
dest_path = '/content/runs/detect/train/weights/best.pt'

"""3.Get a pretrained YOLOV8 model, there are different YOLOV8 models for different task. If already have a trained model, move to step 6

---
Check out the website for more information if neccessary. https://docs.ultralytics.com/models/yolov8/#__tabbed_1_1



"""

model = YOLO('yolov8n.pt')

"""4.Vertify the size of the data"""

# get an image from the dataset
# img = cv.imread('image link')
# print(img.shape)

"""5.Train the pretrained model with the new dataset"""
# use the actual yaml file
result = model.train(data = 'dataset yaml file', epochs = epochs, patience = 50)

"""6.Set the new trained model using the best.pt from the runs folder, or the weights of the computer vision model"""

new_model = YOLO(dest_path)

"""7.Validate the trained model"""

# get the validation result
metrics = new_model.val()

# Print the results
print("Map50 result: ", metrics.box.map50)
print("Map95 result: ", metrics.box.map75)
print("Accuracy of each class: ", metrics.box.maps)

new_model.export(format = 'onnx')