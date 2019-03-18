'''
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
'''
# import the necessary packages
import json
import os
import random
import argparse

import cv2 as cv
import keras.backend as K
import numpy as np
import scipy.io
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image

from car_model import poolerPico
from yolo.yolo_utils import read_classes, read_anchors, preprocess_image
from yolo.yad2k.models.keras_yolo import yolo_head
from yolo.yolo import load_model, yolo_eval, predict
from yolo.retrain_yolo import create_model, draw

# Args
argparser = argparse.ArgumentParser(
    description="Test a pretrained HOV classification pipeline.")

argparser.add_argument(
    '-i',
    '--image',
    help="path to test image",
    default=os.path.join('data', 'test', 'test.jpg'))

argparser.add_argument(
    '-d',
    '--dir',
    help="path to test image directory, with sub directory classifying HOV & no_HOV",
    default=os.path.join('data', 'HOV'))

# constants
cnn_img_width, cnn_img_height = 64, 64
yolo_img_width, yolo_img_height = 416, 416
num_samples = 200
batch_size = 32
check_bumper_sticker = True
score_threshold=0.07
iou_threshold=0.0

'''
Predict based on a CNN model with binary classification
'''
def cnn_predict(filename, labels, model):
    bgr_img = cv.imread(filename)
    bgr_img = cv.resize(bgr_img, (cnn_img_width, cnn_img_height), cv.INTER_CUBIC)
    rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
    rgb_img = np.expand_dims(rgb_img, 0)
  # Predict the image
    pred = model.predict(rgb_img)
    prob = np.max(pred)
    prediction = labels[np.around(prob)]
    text = ('{} predict: {}, prob: {}'.format(filename, prediction, prob))
    #print(text)
    return prediction, prob

'''
Combine preiction of car, persons, bumper, and sticker given an image
'''
def combined_predict(filename, anchors,
                     car_model, car_labels,
                     person_model, person_labels,
                     bumper_model, bumper_labels,
                     sticker_model, sticker_labels):

  # car classification
    prediction, prob = cnn_predict(filename, car_labels, car_model)

  # person detection
    with Image.open(filename) as img:
        image_shape = (float(img.size[1]), float(img.size[0]))

    person_outputs = yolo_head(person_model.output, anchors, len(person_labels))
    boxes, scores, classes = yolo_eval(person_outputs, image_shape)
    sess = K.get_session()
    o_boxes, o_scores, o_classes = predict(sess, filename, scores, boxes, classes,
                                           person_model, person_labels)
    K.clear_session()

    num_persons = len(np.where(o_classes == person_labels.index('person'))[0])

    num_bumpers = 0
    s_prediction = "n/a"
    s_prob = 0.0
    max_prob = 0.0

    if (check_bumper_sticker):
    # bumper detection
      image, image_data = preprocess_image(filename, model_image_size = (yolo_img_width, yolo_img_height))
      draw(bumper_model, bumper_labels, anchors, image_data, image_set='all',
           weights_name='model_data/bumper_yolo.h5', out_path="out", save_all=True)

    return ({'car label': prediction, 'car prob': '{:.4}'.format(prob),
             '2 persons': num_persons >= 2, 'num persons': num_persons,
             'bumper': num_bumpers >= 1, 'num bumpers': num_bumpers,
             'sticker label': s_prediction, 'sticker prob': '{:.4}'.format(max_prob)})
'''
      bumper_outputs = yolo_head(bumper_model.output, anchors, len(bumper_labels))
      image_shape = K.placeholder(shape=(2, ))
      b_boxes, b_scores, b_classes = yolo_eval(bumper_outputs, image_shape,
                                               score_threshold=score_threshold,
                                               iou_threshold=iou_threshold)
      image, image_data = preprocess_image(filename, model_image_size = (yolo_img_width, yolo_img_height))
      sess = K.get_session()
      b_o_boxes, b_o_scores, b_o_classes = sess.run(
          [b_boxes, b_scores, b_classes],
          feed_dict={
              bumper_model.input: image_data,
              image_shape: [image_data.shape[2], image_data.shape[3]],
              K.learning_phase(): 0
          })
      K.clear_session()

      #      b_o_scores, b_o_boxes, b_o_classes = predict(sess, filename,
      #                                             b_scores, b_boxes, b_classes,
      #                                             bumper_model, bumper_labels)

      print(filename, b_o_scores, b_o_boxes, b_o_classes)
      num_bumpers = len(np.where(b_o_classes == bumper_labels.index('bumper'))[0])

      if (num_bumpers >= 1):
        # crop each image, save it to a file
        for i, k in enumerate(np.where(b_o_classes == bumper_labels.index('bumper'))[0]):
          top, left, bottom, right = b_o_boxes[k]
          top = max(0, np.floor(top + 0.5).astype('int32'))
          left = max(0, np.floor(left + 0.5).astype('int32'))
          bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
          right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
          print((left, top), (right, bottom))

          cropped_file = image.crop((left,top,right,bottom)).save(
              os.path.join('out/crop', '{}_{}.png'.format(i,k)))

        # sticker classification
          max_prob = 0.0
          s_prediction, s_prob = cnn_predict(cropped_file, sticker_labels, sticker_model)
          if s_prob > max_prob:
            max_prob = s_prob
    return ({'car label': prediction, 'car prob': '{:.4}'.format(prob),
             '2 persons': num_persons >= 2, 'num persons': num_persons,
             'bumper': num_bumpers >= 1, 'num bumpers': num_bumpers,
             'sticker label': s_prediction, 'sticker prob': '{:.4}'.format(max_prob)})
'''

def _main(args):

  # car classification model, based on an FC-CNN
    car_model, model_name = poolerPico()
    car_model.load_weights('model_data/{}.h5'.format(model_name))
    print("Car classification model")
    car_model.summary()
    car_labels = {0:"no car", 1:"car"}

  # person detection model, based on YOLOv2
    #sess = K.get_session()
    person_labels = read_classes("model_data/object_classes.txt")
    anchors = read_anchors("model_data/yolo_anchors.txt")
    person_model = load_model("model_data/yolo.h5")
    print("Persons detection model")
    person_model.summary()

    bumper_model = None
    bumper_labels = None
    sticker_model = None
    sticker_labels = None
    results = []

    if (check_bumper_sticker):
    # bumper detection model, based on YOLOv2
      bumper_labels = read_classes("model_data/bumper_classes.txt")
      bumper_model_body, bumper_model = create_model(anchors,bumper_labels)
      bumper_model_body.load_weights("model_data/bumper_yolo.h5")
      print("Bumper detection model")
      bumper_model.summary()

    # sticker classification model, based on an FC-CNN
      sticker_model, model_name = poolerPico()
      sticker_model.load_weights('model_data/{}_sticker.h5'.format(model_name))
      print("Sticker classification model")
      sticker_model.summary()
      sticker_labels = {0:"no sticker", 1:"sticker"}

  # Load single image from input path
    result = combined_predict(args.image, anchors,
                              car_model, car_labels,
                              person_model, person_labels,
                              bumper_model_body, bumper_labels,
                              sticker_model, sticker_labels)
    result['file'] = filename
    result['truth'] = 'n/a'
    results.append(result)

  # Load images from input folder with positive images
    for i, image_name in enumerate(os.listdir(args.dir + '/HOV')):
        filename = os.path.join(args.dir, image_name)
        result = combined_predict(args.image, anchors,
                                  car_model, car_labels,
                                  person_model, person_labels,
                                  bumper_model_body, bumper_labels,
                                  sticker_model, sticker_labels)
        result['file'] = filename
        result['truth'] = 'HOV'
        results.append(result)

  # Load images from input folder with negative images
    for i, image_name in enumerate(os.listdir(args.dir + '/noHOV')):
        filename = os.path.join(args.dir, image_name)
        result = combined_predict(args.image, anchors,
                                  car_model, car_labels,
                                  person_model, person_labels,
                                  bumper_model_body, bumper_labels,
                                  sticker_model, sticker_labels)
        result['file'] = filename
        result['truth'] = 'no HOV'
        results.append(result)


    print(results)
#    with open('results.json', 'w') as file:
#        json.dump(results, file, indent=4)


if __name__ == '__main__':
    args = argparser.parse_args()
    _main(args)
