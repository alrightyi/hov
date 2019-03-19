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
import tensorflow as tf
import numpy as np
import scipy.io
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image

from car_model import poolerPico
from yolo.yolo_utils import read_classes, read_anchors, preprocess_image
from yolo.yad2k.models.keras_yolo import yolo_head
from yolo.yolo import load_model, yolo_eval, predict
from yolo.retrain_yolo import create_model
from yolo.yad2k.models.keras_yolo import yolo_eval as retrain_yolo_eval

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
#check_bumper_sticker = True
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
    return prediction, prob

'''
Retrained YOLO predict
'''
def retrain_yolo_predict(model_body, class_names, anchors, image_data,
            weights_name='trained_stage_3_best.h5'):

    image_data = np.array([np.expand_dims(image, axis=0)
        for image in image_data])

    # model.load_weights(weights_name)
    #print(image_data.shape)
    model_body.load_weights(weights_name)

    # Create output variables for prediction.
    yolo_outputs = yolo_head(model_body.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(
        yolo_outputs, input_image_shape, score_threshold=0.07, iou_threshold=0.0)

    # Run prediction on overfit image.
    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

    for i in range(len(image_data)):
        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                model_body.input: image_data[i],
                input_image_shape: [image_data.shape[2], image_data.shape[3]],
                K.learning_phase(): 0
            })
        print('Found {} boxes for image.'.format(len(out_boxes)))
        print(out_boxes)

    return out_boxes, out_scores, out_classes

'''
Combine preiction of car, persons, bumper, and sticker given an image
'''
def combined_predict(filename, anchors,
                     car_graph, car_session, car_model, car_labels,
                     person_graph, person_session, person_model, person_labels,
                     bumper_graph, bumper_session, bumper_model, bumper_labels,
                     sticker_graph, sticker_session, sticker_model, sticker_labels):

    print(filename)
  # car classification
    with car_graph.as_default():
      with car_session.as_default():
        prediction, prob = cnn_predict(filename, car_labels, car_model)

  # person detection
    with person_graph.as_default():
      with person_session.as_default():
        with Image.open(filename) as img:
          image_shape = (float(img.size[1]), float(img.size[0]))

        person_outputs = yolo_head(person_model.output, anchors, len(person_labels))
        boxes, scores, classes = yolo_eval(person_outputs, image_shape)
        o_boxes, o_scores, o_classes = predict(person_session, filename, scores, boxes, classes,
                                               person_model, person_labels)
        num_persons = len(np.where(np.logical_or(o_classes == person_labels.index('person'),
                                                 o_classes == person_labels.index('dog')))[0])

        num_bumpers = 0
        s_prediction = "n/a"
        s_prob = 0.0
        max_prob = 0.0

  # bumper detection
    with bumper_graph.as_default():
      with bumper_session.as_default():
        image, image_data = preprocess_image(filename,
                                             model_image_size = (yolo_img_width, yolo_img_height))
        b_o_boxes, b_o_scores, b_o_classes = retrain_yolo_predict(bumper_model, bumper_labels,
                                                                  anchors, image_data,
                                                                  weights_name='model_data/bumper_yolo.h5')

    #print(filename, b_o_scores, b_o_boxes, b_o_classes)
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

        #print(b_o_boxes[k])
        resized_image = image.resize((yolo_img_height, yolo_img_width), Image.BICUBIC)
        resized_image.crop((left,top,right,bottom)).save(
            os.path.join('out/crop', '{}_{}.png'.format(i,k)))

        # sticker classification
        max_prob = 0.0

        with sticker_graph.as_default():
          with sticker_session.as_default():
            s_prediction, s_prob = cnn_predict('out/crop/{}_{}.png'.format(i,k),
                                               sticker_labels, sticker_model)
        if s_prob > max_prob:
          max_prob = s_prob

    return ({'is car': prediction=='car', 'car prob': '{:.4}'.format(prob),
             '2 persons': num_persons >= 2, 'num persons': num_persons,
             'bumper': num_bumpers >= 1, 'num bumpers': num_bumpers,
             'is sticker': s_prediction=='sticker', 'sticker prob': '{:.4}'.format(max_prob)})


def _main(args):

  # car classification model, based on an FC-CNN
    car_graph = tf.Graph()
    with car_graph.as_default():
      car_session = tf.Session()
      with car_session.as_default():
        car_model, model_name = poolerPico()
        car_model.load_weights('model_data/{}.h5'.format(model_name))
    print("Car classification model")
    car_model.summary()
    car_labels = {0:"no car", 1:"car"}

  # person detection model, based on YOLOv2
    person_graph = tf.Graph()
    with person_graph.as_default():
      person_session = tf.Session()
      with person_session.as_default():
        person_model = load_model("model_data/yolo.h5")
    print("Persons detection model")
    person_model.summary()
    person_labels = read_classes("model_data/object_classes.txt")
    anchors = read_anchors("model_data/yolo_anchors.txt")

    bumper_model = None
    bumper_labels = None
    sticker_model = None
    sticker_labels = None
    results = []

  # bumper detection model, based on YOLOv2
    bumper_graph = tf.Graph()
    with bumper_graph.as_default():
      bumper_session = tf.Session()
      with bumper_session.as_default():
        bumper_labels = read_classes("model_data/bumper_classes.txt")
        bumper_model_body, bumper_model = create_model(anchors,bumper_labels)
        bumper_model_body.load_weights("model_data/bumper_yolo.h5")
    print("Bumper detection model")
    bumper_model.summary()

  # sticker classification model, based on an FC-CNN
    sticker_graph = tf.Graph()
    with sticker_graph.as_default():
      sticker_session = tf.Session()
      with sticker_session.as_default():
        sticker_model, model_name = poolerPico()
        sticker_model.load_weights('model_data/{}_sticker.h5'.format(model_name))
    print("Sticker classification model")
    sticker_model.summary()
    sticker_labels = {0:"no sticker", 1:"sticker"}

    pos_correct = 0
    neg_correct = 0
  # Load single image from input path
    result = combined_predict(args.image, anchors,
                              car_graph, car_session, car_model, car_labels,
                              person_graph, person_session, person_model, person_labels,
                              bumper_graph, bumper_session, bumper_model_body, bumper_labels,
                              sticker_graph, sticker_session, sticker_model, sticker_labels)
    result['file'] = args.image
    result['truth'] = 'n/a'
    result['predict'] = 'HOV' if (result['is car'] and result['2 persons']) or (result['bumper'] and result['is sticker']) else 'noHOV'
    results.append(result)

  # Load images from input folder with positive images
    for i, image_name in enumerate(os.listdir(args.dir + '/HOV')):
      filename = os.path.join(args.dir, 'HOV/{}'.format(image_name))
      result = combined_predict(filename, anchors,
                                car_graph, car_session, car_model, car_labels,
                                person_graph, person_session, person_model, person_labels,
                                bumper_graph, bumper_session, bumper_model_body, bumper_labels,
                                sticker_graph, sticker_session, sticker_model, sticker_labels)
      result['file'] = filename
      result['truth'] = 'HOV'
      result['predict'] = 'HOV' if (result['2 persons']) or (result['bumper'] and result['is sticker']) else 'noHOV'
      results.append(result)
      if result['truth'] == result['predict']:
        pos_correct += 1

  # Load images from input folder with negative images
    for i, image_name in enumerate(os.listdir(args.dir + '/noHOV')):
      filename = os.path.join(args.dir, 'noHOV/{}'.format(image_name))
      result = combined_predict(filename, anchors,
                                car_graph, car_session, car_model, car_labels,
                                person_graph, person_session, person_model, person_labels,
                                bumper_graph, bumper_session, bumper_model_body, bumper_labels,
                                sticker_graph, sticker_session, sticker_model, sticker_labels)

      result['file'] = filename
      result['truth'] = 'noHOV'
      result['predict'] = 'HOV' if (result['2 persons']) or (result['bumper'] and result['is sticker']) else 'noHOV'
      results.append(result)
      if result['truth'] == result['predict']:
        neg_correct += 1

    pos_images = len(os.listdir(args.dir + '/HOV'))
    pos_accuracy = float(pos_correct / pos_images)
    neg_images = len(os.listdir(args.dir + '/noHOV'))
    neg_accuracy = float(neg_correct / neg_images)
    total_images = pos_images + neg_images
    total_accuracy = float((pos_correct+neg_correct) / total_images)

    print('pos images: {}, accuracy: {}'.format(pos_images,pos_accuracy))
    print('neg images: {}, accuracy: {}'.format(neg_images,neg_accuracy))
    print('total images: {}, accuracy: {}'.format(total_images,total_accuracy))
    with open('results.json', 'w') as file:
        json.dump(results, file, indent=4)


if __name__ == '__main__':
    args = argparser.parse_args()
    _main(args)
