# hov
Final project source code for CS230
Compiled and ran on python 3.6.5

Final pipeline for detecting HOV qualification on car images.

Input: Path to a single image and/or path to directory with images separated in two subfolders: HOV, no-HOV.
Output: For single image: HOV or no-HOV based on confidence level of 50% or above.
For folders of iamges, it will give a percentage of correct predictions and output the incorrectly predeicted messages input an output folder.

Steps in the pipeline:
  1) Load/build the models
    a) CNN model for car classification (car / no car)
    b) YOLO model for passenger (person) detection (2)
    c) Retrained YOLO model for bumper detection
    d) Retrained CNN model for clean-air sticker classification (sticker / no sticker)
  2) Preprocess the test image dataset
    - resize
    - normalize
    - human classification
  3) Run prediction logic for each test image:

  if car_classification.predict(image) == car:
    if num_of(passenger_detection(image) == person) >= 2:
      image is HOV qualified
    else if bumper_detection(image) == bumper:
      image = crop(image)
      if sticker_classification(image) == sticker:
        image is HOV qualified

  image is NOT HOV qualified

  4) Compare with human classification and print results


# Installation and instructions
1) Download the source code to your directory.
2) virtualenv -p python3 .env
source .env/bin/activate
pip install -r requirements.txt

# Main script to run:
To run the final pipeline, type: python -m pipeline
This will get the test dataset from ./data/HOV and ./data/test.

# Other executable code I authored/modified:

1) Standalone car classification: python car_model.py
This takes in an image path and determines if the image contains a car or not, based on a pre-trained 3-layer FC-CNN model.

2) Standalone person detection: python yolo/yolo.py
This uses a pre-trained YOLOv2 model of 80 object classes to detect box coordinates of car and persons (among other objects in yolo/model_data/coco_classes.txt).

3) Standalong bumper detection: python yolo/retrain_yolo.py
This script uses transfer learning of an existing YOLOv2 model to retrain for car bumper detection.
It takes in a numpy (npz) file containing dataset images and bounding boxes, and a classes file containing the object class "bumper" and retrains the YOLOv2 model to detect car bumpers.

4) Standalone clean-air sticker classification: python sticker_model.py
This takes a dataset of images and splits them (80/20) into training and verification buckets and trains a 3-layer FC-CNN model to classifiy if the image contains a Californis clean-air sticker (of any type/year).

5) Download and extract bumper labeling on images: yolo/get_labels.py
This script downloads from LabelBox labeled images of car bumpers and puts stores the images and bounding coordinates (boxes) into a numpy (.npz) file for consumption.

# Pre-trained model weights:
- Car classification: model/ppico.h5
- Person detection: model/yolo.h5
- Bumper detection: model/bumper_yolo.h5
- Sticker classification: model/ppico_sticker.h5

# Anchors used to train/run YOLO detection:
- model/yolo_anchors.txt

# Other open-sourced GitHub repos used to construct the pipeline:
https://github.com/foamliu/Car-Recognition.git
https://github.com/alrightyi/stanford_cs230.git (my other repo)
https://github.com/antevis/CarND-Project5-Vehicle_Detection_and_Tracking.git
https://github.com/allanzelener/YAD2K.git
