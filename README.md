# Number Plate Detection and Reading

This repository contains code for a project that involves detecting and reading number plates from images and videos using the YOLO (You Only Look Once) object detection model and Tesseract OCR.

## Introduction

This project utilizes the Ultralytics YOLO library for number plate detection in images and videos. After detection, the Tesseract OCR library is applied to read the text from the detected license plates. The code is structured to process videos and webcam feed as required, and a sample video is included for demonstration purposes.

## Dataset Preparation

The dataset we used for training is in Pascal VOC format and should be converted to YOLO annotation format. The Jupyter notebook `dataset-prep.ipynb` does the required conversion. The code also splits the dataset into train and validation sets. Later more images were added in both the sets from a different dataset. Test set (optional for training) was also created later manually.
To train the YOLO model on your dataset, make sure your directory looks like:

```
...
|-- train/
|   |-- labels/
|   |   |-- image1.txt
|   |   |-- image2.txt
|   |   ...
|   |-- images/
|   |   |-- image1.jpg
|   |   |-- image2.jpg
|   |   ...
|-- val/
|   |-- labels/
|   |   |-- image1.txt
|   |   |-- image2.txt
|   |   ...
|   |-- images/
|   |   |-- image1.jpg
|   |   |-- image2.jpg
|   |   ...
|-- test/               (optional)
|   |-- labels/
|   |   |-- image1.txt
|   |   |-- image2.txt
|   |   ...
|   |-- images/
|   |   |-- image1.jpg
|   |   |-- image2.jpg
|   |   ...
|-- dataset.yaml
```
The YAML file should contain the paths of the train, test, and validation sets and a mapping of labels to class names.

## Training
The Jupyter notebook `train.ipynb` trains the YOLO model on our custom dataset which has been prepared in the required format.

## License Plate Detection in Videos

The main script (`yolo.py`) includes a loop for processing video frames in real-time. It reads in the input video or incoming feed from the webcam frame by frame, detects any licence plates. On successful detection, the licence plate is cropped and fed into tesseract-ocr which reads the number plates. This can be then cross checked with any database and then appropriate action can be taken for the cars whose licence plate numbers have been registered.
