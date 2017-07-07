import cv2
import csv
import numpy as np
from scipy import ndimage
from sklearn.model_selection import train_test_split

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 160
IMAGE_CHANNELS = 3

def random_shadow(image):
    '''
    Randomly add some shadow in a part of the image
    '''
    # Using HLS(Hue, Light, Saturation) representation makes it easy
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    # Copy the HLS array to have the save size.
    mask = 0 * hls[:,:,1]
    # Pick 2 ramdom points on the upper and lower edges of the image
    # to trace a straight line between the 2.
    topX = np.random.uniform() * IMAGE_WIDTH
    topY = 0
    bottomX = np.random.uniform() * IMAGE_WIDTH
    bottomY = IMAGE_HEIGHT
    # A mgrid would help decide whether a pixel is left or right of that line
    mX, mY = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

    mask[(mY - topY) * (bottomX - topX) - (bottomY - topY) * (mX - topX) >= 0] = 1
    side = mask == np.random.randint(2)
    shadow_ratio = np.random.uniform(low=0.2, high=0.5)
    # Diminish the saturation component of the HLS image
    hls[:, :, 1][side] = hls[:, :, 1][side] * shadow_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

def random_brightness(image):
    '''
    Randomly increase or decrease brightness of the whole image.
    '''
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 0.5 + np.random.uniform()
    hsv[:,:,2] = hsv[:,:,2] * ratio
    hsv[:,:,2][hsv[:,:,2]>255.] = 255.
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def random_shift(image, steering_angle):
    '''
    Shifts the image horizontally and vertically
    '''
    X_range = 100
    Y_range = 20
    dX = X_range * np.random.uniform() - X_range / 2
    steering_angle += dX/X_range * .4
    dY = Y_range * np.random.uniform() - Y_range / 2
    shift = np.float32([[1,0,dX],[0,1,dY]])
    image = cv2.warpAffine(image,shift,(image.shape[1], image.shape[0]))
    return image, steering_angle

def load_paths_labels(training_images_folder):
    '''
    Load the relative paths of the 3 images and steering angle for each line
    '''
    lines = []
    with open(training_images_folder + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    image_paths = []
    steering_angles = []
    for line in lines:
        center, left, right = line[0], line[1], line[2]
        center = training_images_folder + 'IMG/' + center.split('/')[-1]
        left = training_images_folder + 'IMG/' + left.split('/')[-1]
        right = training_images_folder + 'IMG/' + right.split('/')[-1]
        image_paths.append((center, left, right))
        steering_angle = float(line[3])
        steering_angles.append(steering_angle)
    return np.array(image_paths), np.array(steering_angles)

def load_N_split(parameter):
    '''
    Load images paths, their respective steering_angles and split them
    in a training set and a validation set
    '''
    image_paths , steering_angles = load_paths_labels(parameter['training_images_folder'])
    return train_test_split(image_paths, steering_angles, test_size=parameter['valid_over_train_ratio'], random_state=0)

def load_image(path):
    '''
    Read an image in RGB format
    '''
    return ndimage.imread(path)

def RGB2YUV(image):
    '''
    Conversion from RGB to YUV representation as used by Nvidia
    '''
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

def preprocess(image):
    '''
    Preprocessing pipeline
    Cropping done with a keras call.
    '''
    image = RGB2YUV(image)
    return image

def random_image_pick(center, left, right, steering_angle, bias):
    '''
    Randomly chooses an image between the 3 available
    The center camera image has only half a chance to be chosen
    if its steering_angle is among [-0.5, 0.5]
    '''
    if abs(steering_angle) <= 0.00 and np.random.choice(2) == 0:
        pick = np.random.choice(3)
        if pick == 1:
            return load_image(center), steering_angle
        elif pick == 2:
            return load_image(left), steering_angle + bias
        else:
            return load_image(right), steering_angle - bias
    else:
        pick = np.random.choice(2)
        if pick == 1:
            return load_image(right), steering_angle - bias
        else:
            return load_image(left), steering_angle + bias

    # pick = np.random.uniform()
    # probs = [abs(steering_angle - bias), abs(steering_angle), abs(steering_angle + bias)]
    # probs /= sum(probs)
    # if pick < probs[0]:
    #     return load_image(right), steering_angle - bias
    # elif pick < probs[1]:
    #     return load_image(center), steering_angle
    # else:
    #     return load_image(left), steering_angle + bias


def horizontal_flip(image, steering_angle):
    '''
    Flips the image horizontally and well as the steering_angle
    to balance the dataset
    '''
    return cv2.flip(image, 1), steering_angle * -1.

def augment(center, left, right, steering_angle, parameter):
    '''
    Augment the data to improve the model accuracy and prevent overfitting
    '''
    # The steering_angle is the one from the center image so far
    image , steer = random_image_pick(center, left, right, steering_angle, parameter['steering_bias'])
    # Now, the steering_angle is adapted to the chosen image
    if np.random.choice(2) == 0:
        image, steer = horizontal_flip(image, steer)
    if parameter['color_augmentation']:
        image = random_brightness(image)
        image = random_shadow(image)
    if parameter['shift_augmentation']:
        image, steer = random_shift(image, steer)
    return image, steer

def batch_generator(image_paths, steering_angles, parameter, is_training):
    '''
    Provide batches of preprocessed images and their respective steering angles
    '''
    images = np.zeros([parameter['batch_size'], IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steering_angles_out = np.zeros(parameter['batch_size'])
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[index]
            steering_angle = steering_angles[index]
            if is_training:# and np.random.rand() < 0.75:
                image, steering_angle = augment(center, left, right, steering_angle, parameter)
            else:
                image = load_image(center)
            # add the image and steering angle to the batch
            images[i] = preprocess(image)
            steering_angles_out[i] = steering_angle
            i += 1
            if i == parameter['batch_size']:
                break
        yield images, steering_angles_out


###################
##      MEMO     ##

# np.mgrid[0:5,0:5]
# array([[[0, 0, 0, 0, 0],
#     [1, 1, 1, 1, 1],
#     [2, 2, 2, 2, 2],
#     [3, 3, 3, 3, 3],
#     [4, 4, 4, 4, 4]],
#
#    [[0, 1, 2, 3, 4],
#     [0, 1, 2, 3, 4],
#     [0, 1, 2, 3, 4],
#     [0, 1, 2, 3, 4],
#     [0, 1, 2, 3, 4]]])
