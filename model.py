import csv
import cv2
import numpy as np
from random import *

# Parameters
STEERING_OFFSET=0.2
VALIDATION_PERCENTAGE=0.2
EPOCHS=1

# Folders to check for training data
TRAINING_DATA = ['regular_rounds', 'last_curve', 'training1', 'training2', 'training3', 'training4', 'training7', 'training8']

# ColumnIds for the testdata
IMG_CENTER = 0
IMG_LEFT = 1
IMG_RIGHT = 2
STEERING_ANGLE = 3
THROTTLE = 4
BREAK = 5
SPEED = 6

import sklearn

def generator(lines, batch_size=36):
    len_lines = len(lines)
    while 1: # loop never stops so generator never stops
        lines = sklearn.utils.shuffle(lines)
        for offset in range(0, len_lines, batch_size):
            batch_lines = lines[offset:offset+batch_size]
            images = []
            steering_angles = []
            for line in batch_lines:
                # center image
                images.append(loadImage(line, IMG_CENTER))
                steering_angle = float(line[STEERING_ANGLE])
                speed = float(line[SPEED])
                # adjusting steering angle to speed, test data 30mph, autonomus driving 9 mph
                steering_angles.append(steering_angle)
                # left image + correction of stearing
                images.append(loadImage(line, IMG_LEFT))
                steering_angles.append(steering_angle + STEERING_OFFSET)
                # left image + correction of stearing
                images.append(loadImage(line, IMG_RIGHT))
                steering_angles.append(steering_angle - STEERING_OFFSET)

            # augmentation
            augmented_images = []
            augmented_steering_angles = []
            for image, steering_angle in zip(images, steering_angles):
                # flipping
                augmented_images.append(cv2.flip(image,1))
                augmented_steering_angles.append(steering_angle*-1.0)

            images.extend(augmented_images)
            steering_angles.extend(augmented_steering_angles)
            inputs = np.array(images)
            outputs = np.array(steering_angles)
            yield sklearn.utils.shuffle(inputs, outputs)

def loadImage(line, column):
    image = cv2.imread(line[column])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Loads the training data from the provided folder paramter
def loadLines(folder):
    print("Loading training data from folder: %s" % folder)
    lines = []
    with open('./' + folder + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            line[IMG_CENTER] = fixPath(folder, line[IMG_CENTER])
            line[IMG_LEFT] = fixPath(folder, line[IMG_LEFT])
            line[IMG_RIGHT] = fixPath(folder, line[IMG_RIGHT])
            lines.append(line)

    return lines

# Get image from line and folder and column (CENTER, LEFT, RIGHT)
def fixPath(folder, path):
    filename = path.split('/')[-1]
    return './' + folder + '/IMG/' + filename

lines = []
# Load test data from each training folder
for folder in TRAINING_DATA:
    lines.extend(loadLines(folder))
print("Loaded {} lines of testdata".format(len(lines)))

# Splitting samples and creating generators.
from sklearn.model_selection import train_test_split
train_lines, validation_lines = train_test_split(lines, test_size=0.2)

print('Train lines: {}'.format(len(train_lines)))
print('Validation lines: {}'.format(len(validation_lines)))

train_generator = generator(train_lines)
validation_generator = generator(validation_lines)

def nVidiaModel():
    from keras.models import Sequential
    from keras.layers import Cropping2D, Lambda, Flatten, Dense, MaxPooling2D, Convolution2D

    model = Sequential()
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

model = nVidiaModel()
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, \
                    samples_per_epoch=len(train_lines) * 6, \
                    validation_data=validation_generator, \
                    nb_val_samples=len(validation_lines) * 6, \
                    nb_epoch=EPOCHS, \
                    verbose=1)

model.save('model.h5')
