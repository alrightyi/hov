from keras.layers import Conv2D, Flatten, Lambda, MaxPooling2D, Dropout
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from pandas import DataFrame
#from resnet_152 import resnet152_model

import helper as aux
import glob
from sklearn.utils import shuffle
import numpy as np
from sklearn.model_selection import train_test_split as trainTestSplit
import pickle
import os
from keras.callbacks import ModelCheckpoint

width, height, depth = (64,64,3)

# Fully convolutional neural network model
def poolerPico(object_type, inputShape=(width,height,depth)):
    """
    So-called 'Fully-convolutional Neural Network' (FCNN). Single filter in the top layer
    used for binary classification of 'object/no_object'
    :param inputShape: 
    :return: Keras model, model name
    """
    model = Sequential()
    # Center and normalize our data
    #model.add(Lambda(lambda x: x / 255., input_shape=inputShape, output_shape=inputShape))
    # Block 0
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', name='cv0',
                     input_shape=inputShape, padding="same"))
    model.add(Dropout(0.5))

    # Block 1
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', name='cv1', padding="same"))
    model.add(Dropout(0.5))

    # block 2
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', name='cv2', padding="same"))
    model.add(MaxPooling2D(pool_size=(8, 8)))
    model.add(Dropout(0.5))

    # binary 'classifier'
    model.add(Conv2D(filters=1, kernel_size=(8, 8), name='fcn', activation="sigmoid"))

    return model, 'ppico_{}'.format(object_type)


def generator(samples, batchSize=32, useFlips=False, resize=False):
    """
    Generator to supply batches of sample images and labels
    :param samples: list of sample images file names
    :param batchSize: 
    :param useFlips: adds horizontal flips if True (effectively inflates training set by a factor of 2)
    :param resize: Halves images widths and heights if True
    :return: batch of images and labels
    """
    samplesCount = len(samples)

    while True:  # Loop forever so the generator never terminates
        shuffle(samples, random_state=42)
        for offset in range(0, samplesCount, batchSize):
            batchSamples = samples[offset:offset + batchSize]

            xTrain = []
            yTrain = []
            for batchSample in batchSamples:
                y = float(batchSample[1])

                fileName = batchSample[0]
                #print(fileName)
                #print('data/sticker/sticker/30. stickers.jpg')
                #image = aux.rgbImage('data/sticker/sticker/30. stickers.jpg', resize=resize)
                image = aux.rgbImage(fileName, resize=resize)

                xTrain.append(image)
                yTrain.append(y)

                if useFlips:
                    flipImg = aux.flipImage(image)
                    xTrain.append(flipImg)
                    yTrain.append(y)

            xTrain = np.array(xTrain)
            yTrain = np.expand_dims(yTrain, axis=1)

            yield shuffle(xTrain, yTrain, random_state=42)  # Since we added flips, better shuffle again


def createSamples(x, y):
    """
    Returns a list of tuples (x, y)
    :param x: 
    :param y: 
    :return: 
    """
    assert len(x) == len(y)

    return [(x[i], y[i]) for i in range(len(x))]


def getData(object_type='sticker'):
    """
    Creates dataset where x are image file names, y - labels (0 for non_object / 1 for object)
    :return: 
    """
    dataFile = 'data/{}/data.p'.format(object_type)

    if not os.path.isfile(dataFile):
        tryGenerateNew = aux.promptForInputCategorical(message='data file not found. Attempt to generate?',
                                                       options=['y', 'n']) == 'y'

        if tryGenerateNew:
            objectFolder = 'data/{}/{}'.format(object_type,object_type)
            noObjectFolder = 'data/{}/no_{}'.format(object_type,object_type)

            if not os.path.isdir(objectFolder) or not os.path.isdir(noObjectFolder):
                print('No samples found.')
                return None, None, None, None, None, None
            else:
                objectFiles = glob.glob('{}*/*.*'.format(objectFolder), recursive=True)
                noObjectFiles = glob.glob('{}*/*.*'.format(noObjectFolder), recursive=True)

                imageSamplesFiles = objectFiles + noObjectFiles
                #y = np.concatenate((np.ones(len(objectFiles)), np.zeros(len(noObjectFiles))))
                pos = np.chararray((1,len(objectFiles)),itemsize=10)
                pos[:] = str(object_type)
                print(pos)
                neg = np.chararray((1,len(noObjectFiles)),itemsize=10)
                neg[:] = 'no_{}'.format(str(object_type))
                y = np.concatenate((pos,neg))
                print(y)
                print('objectFile size = ' + str(len(objectFiles)) + ', noObjectFile size = ' + str(len(noObjectFiles)))

                imageSamplesFiles, y = shuffle(imageSamplesFiles, y, random_state=42)

                # Using skLearn utils to split data to train and test sets
                xTrain, xTest, yTrain, yTest = trainTestSplit(imageSamplesFiles, y, test_size=0.2, random_state=42)

                # Further split train data to train and validation
                xTrain, xVal, yTrain, yVal = trainTestSplit(xTrain, yTrain, test_size=0.2, random_state=42)

                data = {'xTrain': xTrain, 'xValidation': xVal, 'xTest': xTest,
                        'yTrain': yTrain, 'yValidation': yVal, 'yTest': yTest}

                pickle.dump(data, open(dataFile, 'wb'))

                return xTrain, xVal, xTest, yTrain, yVal, yTest

        else:
            return None, None, None, None, None, None
    else:
        with open(dataFile, mode='rb') as f:
            data = pickle.load(f)

            xTrain = data['xTrain']
            xValidation = data['xValidation']
            xTest = data['xTest']
            yTrain = data['yTrain']
            yValidation = data['yValidation']
            yTest = data['yTest']

            return xTrain, xValidation, xTest, yTrain, yValidation, yTest


def main():
    '''
    xTrain, xVal, xTest, yTrain, yVal, yTest = getData()

    #trainSamples = createSamples(xTrain, yTrain)
    trainSamples = {'x': xTrain, 'y': yTrain}
    trainDf = DataFrame(trainSamples)
    print(trainDf)
    #validationSamples = createSamples(xVal, yVal)
    validationSamples = {'x':xVal, 'y':yVal}
    validDf = DataFrame(validationSamples)
    print(validDf)
    print('trainSamples: ' + str(len(xTrain)) + ', valSamples: ' + str(len(xVal)))

    # batchSize = 32
    # useFlips = True
    # epochCount = 3
    '''
    batchSize = aux.promptForInt(message='Please specify the batch size (32, 64, etc.): ')
    #useFlips = aux.promptForInputCategorical('Use flips?', options=['y', 'n']) == 'y'
    epochCount = aux.promptForInt(message='Please specify the number of epochs: ')
    '''
    inflateFactor = 8 if useFlips else 1

    # Keras generator params computation
    stepsPerEpoch = len(xTrain) * inflateFactor / batchSize
    print('steps per epoch: {}'.format(stepsPerEpoch))

    validationSteps = len(xVal) * inflateFactor / batchSize
    print('validation steps per epoch: {}'.format(validationSteps))
    '''
    proceed = aux.promptForInputCategorical('Proceed?', ['y', 'n']) == 'y'

    if proceed:

        sourceModel, modelName = poolerPico('sticker')
        #sourceModel = resnet152_model(height, width, color_type=depth, num_classes=2)

        # Adding fully-connected layer to train the 'classifier'
        x = sourceModel.output
        x = Flatten()(x)
        model = Model(inputs=sourceModel.input, outputs=x)
        #model = sourceModel

        print(model.summary())

        confirm = aux.promptForInputCategorical('Confirm?', ['y', 'n']) == 'y'

        if not confirm:
            return
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        # Instantiating train and validation generators
        #trainGen = generator(samples=trainSamples, useFlips=useFlips, resize=True)
        generator = ImageDataGenerator(
            #featurewise_center=True,
            #featurewise_std_normalization=True,
            samplewise_center=True,
            samplewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            brightness_range=[0.75,1.25],
            channel_shift_range=1.2,
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            validation_split=0.2,
            horizontal_flip=True)
        #trainGen.fit(xTrain)
        trainGenerator = generator.flow_from_directory(
            'data/sticker',
            #dataframe=trainDf,
            #x_col='x',
            #y_col='y',
            target_size=(width,height),
            batch_size=batchSize,
            class_mode='binary',
            subset='training')
        #validGen = generator(samples=validationSamples, useFlips=useFlips, resize=True)
        #validGen = ImageDataGenerator(
        #    rescale=1./255)
        validGenerator = generator.flow_from_directory(
            'data/sticker',
            #dataframe=validDf,
            #x_col='x',
            #y_col='y',
            target_size=(width,height),
            batch_size=batchSize,
            class_mode='binary',
            #save_to_dir='data/augment_sticker',
            subset='validation')

        timeStamp = aux.timeStamp()
        weightsFile = '{}_{}.h5'.format(modelName, timeStamp)

        checkpointer = ModelCheckpoint(filepath=weightsFile,
                                       monitor='val_acc', verbose=0, save_best_only=True)

        _ = model.fit_generator(trainGenerator,
                                steps_per_epoch=trainGenerator.samples/batchSize,
                                #stepsPerEpoch,
                                validation_data=validGenerator,
                                validation_steps=validGenerator.samples/batchSize,
                                #validationSteps,
                                epochs=epochCount, callbacks=[checkpointer])

        print('Training complete. Weights for best validation accuracy have been saved to {}.'
              .format(weightsFile))
        '''
        # Evaluating accuracy on test set
        print('Evaluating accuracy on test set.')
        #testSamples = createSamples(xTest, yTest)
        testSamples = {'x':xTest, 'y':yTest}
        testDf = DataFrame(testSamples)
        #testGen = generator(samples=testSamples, useFlips=False, resize=True)
        testGen = ImageDataGenerator(
            rescale=1./255)
        testGenerator = testGen.flow_from_dataframe(
            dataframe=testDf,
            x_col='x',
            y_col='y',
            target_size=(64,64),
            batch_size=batchSize,
            class_mode='binary')

        testSteps = len(testSamples) / batchSize
        accuracy = model.evaluate_generator(generator=testGenerator, steps=testSteps)

        print('test accuracy: ', accuracy)
        '''

if __name__ == '__main__':
    main()
