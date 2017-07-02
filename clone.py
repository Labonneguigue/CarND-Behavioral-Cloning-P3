import os
import csv
import cv2
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model, model_from_json
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Dropout
#from keras.utils.visualize_util import plot
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

import gc
#check followings
#from utils import INPUT_SHAPE, batch_generator

##      IDEAS       ##

#1. Randomise image brightness
#2. Crop camera
#3. Type of model : Nvidia model, LeNet, Comma.ai ..
#4. Remove images/rows when driving straight
#5. Camera rotation
#6. Early Stopping
#7. Using 3 cameras

# TODO
# Check what is from keras.callbacks import TensorBoard, ModelCheckpoint


##      Config      ##
saved_model = 'model.h5'
saved_images_folder = './saved_images/'
training_images_folder = '../data/driving_data_mouse/'
######################
print("Saving model to file : " + saved_model)
print("Saving created images to folder : " + saved_images_folder)
print("Retrieving training images and labeling data from csv file : " + training_images_folder)

##      Parameters      ##
# 'model_type' is either "commaai" or "nvidia"

parameter = {'dropout' : 0.5,
             'model_type' : '',
             'epochs' : 1,
             'loss_function' : 'mse',
             'optimizer' : 'adam',
             'reload_model' : 1}

##########################
import os.path

if parameter['reload_model']:
    if os.path.isfile('./model.json'):
        # load json and create model
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("weights.h5")
        print("Loaded model from disk")

print("Model selected : " + parameter['model_type'])
print("Dropout applied : {}".format(parameter['dropout']))



lines = []
with open(training_images_folder + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = training_images_folder + 'IMG/' + filename
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


X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

earlyStopping = EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')

def build_basic_test_model():
    model = Sequential()
    model.add(Flatten(input_shape=(160,320,3)))
    model.add(Dense(1))
    model.summary()
    return model

def build_commaai_model(**argv):
	"""
	Creates the comma.ai model, and returns a reference to the model
	The comma.ai model's original source code is available at:
	https://github.com/commaai/research/blob/master/train_steering_model.py
	"""
	ch, row, col = CH, H, W  # camera format

	model = Sequential()
	model.add(Lambda(lambda x: x/255. - 0.5, input_shape=(160, 320,3)))
	model.add(Convolution2D(16, 8, 8, activation='elu', subsample=(4, 4), border_mode='same', W_regularizer=l2(L2_REG_SCALE)))
	model.add(Convolution2D(32, 5, 5, activation='elu', subsample=(2, 2), border_mode='same', W_regularizer=l2(L2_REG_SCALE)))
	model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode='same', W_regularizer=l2(L2_REG_SCALE)))
	model.add(Flatten())
	model.add(Dropout(.2))
	model.add(ELU())
	model.add(Dense(512, W_regularizer=l2(0.)))
	model.add(Dropout(.5))
	model.add(ELU())
	model.add(Dense(1, W_regularizer=l2(0.)))
	#model.compile(optimizer=Adam(lr=LR), loss='mean_squared_error')
    model.summary()
    return model

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

if parameter['model_type'] == 'nvidia':
    print("nvidia")
    model = build_nvidia_model(**parameter)
elif parameter['model_type'] == 'commaai':
    print("commaai")
    model = build_commaai_model(**parameter)
else:
    model = build_basic_test_model()
    print("Basic model.")


#plot(model, to_file=saved_images_folder+'model.png', show_shapes=True)

# Regression network so we use the mean squared error != cross entropy -> classification network
model.compile(loss=parameter['loss_function'], optimizer=parameter['optimizer'])
history_object = model.fit(X_train, y_train,
                            validation_split=0.2,
                            shuffle=True,
                            nb_epoch=parameter['epochs'],
                            callbacks=[earlyStopping],
                            verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


model.save(saved_model)
print("Model saved to file : " + saved_model)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("weights.h5")
print("Saved model to disk")

exit()
gc.collect()
