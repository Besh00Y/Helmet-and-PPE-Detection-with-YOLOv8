# Helmet-and-PPE-Detection-with-YOLOv8

This project focuses on detecting helmets and personal protective equipment (PPE) using the YOLOv8 object detection model. It utilizes two datasets for training: one for helmets and another for full PPE, both obtained through Roboflow.

## Project Overview
The goal of this project is to train a YOLOv8 model to accurately detect helmets and PPE in images or videos. Two datasets were used for training:

## Helmet Detection Dataset: 
A dataset to detect whether individuals are wearing helmets.
PPE Detection Dataset: A more comprehensive dataset for detecting various forms of PPE, including helmets, vests, etc.
The model is trained using the YOLOv8 architecture and fine-tuned on both datasets for accurate detection.

## Datasets

Helmet Detection Dataset:
Source: Roboflow
Task: Detect the presence of helmets on individuals.
Dataset ID: helmet-uu1my-rm8bt
Version: 3

PPE Detection Dataset:

Source: Roboflow
Task: Detect various PPE like helmets, vests, and other protective equipment.
Dataset ID: synapsis-ppe
Version: 513

## Dataset Download Instructions
Both datasets are hosted on Roboflow. To download them, you need to authenticate via the Roboflow API.
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")

# Helmet Detection Dataset
project = rf.workspace("04rladusdn").project("helmet-uu1my-rm8bt")
version = project.version(3)
dataset = version.download("yolov8")

# PPE Detection Dataset
project = rf.workspace("computer-vision-8kpih").project("synapsis-ppe")
version = project.version(513)
dataset = version.download("yolov8")
Replace "YOUR_API_KEY" with your actual Roboflow API key.

## Training the Model
The model is trained on the YOLOv8 architecture, utilizing two different datasets to detect helmets and PPE. The training process uses the following setup:

Model: YOLOv8n (YOLOv8 nano model for fast detection)
Image Size: 640x640
Epochs: 30 for helmet detection, 40 for PPE detection
Helmet Detection Training
!yolo task=detect mode=train model=yolov8n.pt data=/content/helmet-3/data.yaml epochs=30 imgsz=640 plots=True

PPE Detection Training

!yolo task=detect mode=train model=yolov8n.pt data=/content/synapsis-ppe-513/data.yaml epochs=40 imgsz=640 plots=True
Key Training Parameters:
task: detect (object detection task)
mode: train (training mode)
model: yolov8n.pt (nano version of YOLOv8 for lightweight detection)
data: Path to the dataset YAML configuration file
epochs: Number of training epochs
imgsz: Image size for input
plots: Generates training plots for visualizing the learning process
Model Evaluation
After training, the model is evaluated based on precision, recall, mAP (mean average precision), and F1 score. You can use the evaluation plots to monitor the model’s performance during training.

## Example Results:
mAP@50: This is a key metric that evaluates the model's detection accuracy at an Intersection over Union (IoU) threshold of 50%.
Precision & Recall: These metrics help assess the model’s performance in detecting helmets and PPE.
Running Inference
After training, the YOLOv8 model can be used to detect helmets and PPE in new images or videos. You can load the trained model and pass it through images or video streams for real-time detection.

## Example Usage:
from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO('best.pt')

# Load an image
image = cv2.imread('path_to_image.jpg')

# Perform detection
results = model(image)

# Visualize results
results.show()

## Installation Instructions

Install dependencies:

pip install ultralytics roboflow opencv-python
Download the datasets: Follow the instructions above to download the helmet and PPE datasets using Roboflow.

Train the model: After downloading the datasets, train the YOLOv8 model using the commands provided for helmet and PPE detection.

## Future Improvements
Data Augmentation: Implement additional data augmentation techniques to improve generalization.
Fine-tuning: Experiment with different YOLOv8 variants (e.g., yolov8m, yolov8l) for more accurate detection.
Real-time Deployment: Integrate real-time video feed for helmet and PPE detection in industrial environments.

## Conclusion
This project demonstrates the ability to detect helmets and PPE using the YOLOv8 object detection model. With sufficient training data, the model performs well and can be deployed for safety compliance monitoring in industrial settings.
