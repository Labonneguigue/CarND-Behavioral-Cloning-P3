import os
import csv
import cv2
import json
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import gc
from keras.models import Sequential, Model, model_from_json
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Dropout, ELU
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.regularizers import l2
# data.py provides the image generator and the augmentation functions.
from data import batch_generator, load_N_split


##      IDEAS       ##

#1. Randomise image brightness
#2. Crop camera
#3. Type of model : Nvidia model, LeNet, Comma.ai ..
#4. Remove images/rows when driving straight
#5. Camera rotation
#6. Early Stopping
#7. Using 3 cameras


##      Parameters      ##
# 'model_type' is either "commaai" or "nvidia"

parameter = {'dropout' : 0.5,
             'model_type' : 'nvidia',
             'loss_function' : 'mse',
             'optimizer' : 'adam',
             'ESpatience' : 0,
             'reload_model' : 0,
             'steering_bias' : 0.25,
             'valid_over_train_ratio' : 0.15,
             'batch_size' : 128,
             'samples_per_epochs' : 40 * 128,
             'epochs' : 5,
             'color_augmentation' : 0,
             'shift_augmentation' : 1,
             'true_random_pick' : 0,
             'saved_model' : './models/model.h5',
             'saved_images_folder' : './saved_images/',
             'training_images_folder' : '../data/BehaviorCloning/mouse_dataset/'}
             #'training_images_folder' : '../data/BehaviorCloning/extra/'}

##########################


def old_load_data():
    lines = []
    with open(parameter['training_images_folder'] + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    images = []
    measurements = []
    for line in lines:
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = parameter['training_images_folder'] + 'IMG/' + filename
        image = ndimage.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

    augmented_images, augmented_measurements = [], []
    for image, measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        augmented_images.append(cv2.flip(image, 1))
        augmented_measurements.append(measurement * -1.0)

    return np.array(augmented_images), np.array(augmented_measurements)

'''
Declares a callback function called at the end of each epoch.
Stores the model weights and restores an previous version if the validation loss increases
'''
earlyStopping = EarlyStopping(monitor='val_loss', patience=parameter['ESpatience'], verbose=1, mode='auto')

'''
Declares a callback for logging all information relative to TensorBoard for later visualization
'''
tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

def build_basic_test_model():
    '''
    Builds a basic model for testing
    '''
    model = Sequential()
    model.add(Lambda(lambda x: x/255. - 0.5, input_shape=(160, 320,3)))
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    model.add(Flatten())
    model.add(Dense(1))
    model.summary()
    return model

def build_commaai_model(**argv):
    """
    Creates the comma.ai model, and returns a reference to the model
    The comma.ai model's original source code is available at:
    https://github.com/commaai/research/blob/master/train_steering_model.py
    """
    model = Sequential()
    model.add(Lambda(lambda x: x/255. - 0.5, input_shape=(160, 320,3)))
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    model.add(Conv2D(16, 8, 8, activation='elu', subsample=(4, 4), border_mode='same', W_regularizer=l2(0.)))
    model.add(Conv2D(32, 5, 5, activation='elu', subsample=(2, 2), border_mode='same', W_regularizer=l2(0.)))
    model.add(Conv2D(64, 5, 5, subsample=(2, 2), border_mode='same', W_regularizer=l2(0.)))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512, W_regularizer=l2(0.)))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1, W_regularizer=l2(0.)))
    model.summary()
    return model
    #model.compile(optimizer=Adam(lr=LR), loss='mean_squared_error')

def build_nvidia_model(**argv):
    """
    Model developed by NVIDIA - adapted to my images.
    """
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320,3)))
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Dropout(argv["dropout"]))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()
    return model


if __name__ == "__main__":

    ##      Config      ##
    print("Saving model to file : " + parameter['saved_model'])
    print("Saving created images to folder : " + parameter['saved_images_folder'])
    print("Retrieving training images and labeling data from csv file : " + parameter['training_images_folder'])
    ######################

    isModelLoaded = 0
    if parameter['reload_model']:
        if os.path.isfile('./models/model.json'):
            # load json and create model
            json_file = open('./models/model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights("./models/weights.h5")
            isModelLoaded = 1
            print("Loaded model from disk")
        else:
            print("No model previously stored. Building new one.")
    else:
        print("Building new one.")

    print("Model selected : " + parameter['model_type'])
    print("Dropout applied : {}".format(parameter['dropout']))
    print("Extracting data ... ")

    #X_train = np.array(augmented_images)
    #y_train = np.array(augmented_measurements)
    #X_train, y_train = old_load_data()


    image_paths_train, image_paths_valid, steering_angles_train, steering_angles_valid = load_N_split(parameter)

    print("Data extracted.")
    print("Training data")
    print(image_paths_train.shape)
    print("validation data")
    print(image_paths_valid.shape)
    assert(image_paths_train.shape[0] == steering_angles_train.shape[0])
    assert(image_paths_valid.shape[0] == steering_angles_valid.shape[0])

    if isModelLoaded:
        model = loaded_model

    else:
        if parameter['model_type'] == 'nvidia':
            print("Creating model Nvidia")
            model = build_nvidia_model(**parameter)
        elif parameter['model_type'] == 'commaai':
            print("Creating model Comma.ai")
            model = build_commaai_model(**parameter)
        else:
            print("Creating a Basic model.")
            model = build_basic_test_model()

        #plot(model, to_file=saved_images_folder+'model.png', show_shapes=True)
        print("Model created. Compiling ...")

    # Regression network so we use the mean squared error != cross entropy -> classification network
    model.compile(loss=parameter['loss_function'], optimizer=parameter['optimizer'])

    print("Model compiled. Training ...")

    #history_object = model.fit(X_train, y_train,
                                # validation_split=0.2,
                                # shuffle=True,
                                # nb_epoch=parameter['epochs'],
                                # callbacks=[earlyStopping],
                                # verbose=1)

    history_object = model.fit_generator(batch_generator(image_paths_train,
                                                        steering_angles_train,
                                                        parameter, True),
                                        parameter['samples_per_epochs'],
                                        nb_epoch=parameter['epochs'],
                                        max_q_size=1,
                                        validation_data=batch_generator(image_paths_valid,
                                                                        steering_angles_valid,
                                                                        parameter, False),
                                        nb_val_samples=image_paths_train.shape[0],
                                        callbacks=[earlyStopping, tbCallBack],
                                        verbose=1)


    model.save(parameter['saved_model'])
    print("Model saved to file : " + parameter['saved_model'])

    # serialize model to JSON
    model_json = model.to_json()
    with open("./models/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize parameters
    # serialize weights to HDF5
    model.save_weights("./models/weights.h5")
    print("Saved model to disk")

    quit()
    gc.collect()
