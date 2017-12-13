import csv
import cv2
import numpy as np

# Parameters
STEERING_OFFSET=0.2
VALIDATION_PERCENTAGE=0.2
EPOCHS=10

# Folders to check for training data
TRAINING_DATA = ['training1', 'training2', 'training3', 'training4', 'training5']

# ColumnIds for the testdata
IMG_CENTER = 0
IMG_LEFT = 1
IMG_RIGHT = 2
STEERING_ANGLE = 3
THROTTLE = 4
BREAK = 5
SPEED = 6

# Loads the training data from the provided folder paramter
def loadImages(folder):
    print("Loading training data from folder: %s" % folder)
    lines = []
    with open('./' + folder + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    images = []
    steering_angles = []
    for line in lines:
        # center image
        images.append(getImage(line, folder, IMG_CENTER))
        steering_angle = float(line[STEERING_ANGLE])
        steering_angles.append(steering_angle)
        # left image + correction of stearing
        images.append(getImage(line, folder, IMG_LEFT))
        steering_angles.append(steering_angle + STEERING_OFFSET)
        # left image + correction of stearing
        images.append(getImage(line, folder, IMG_RIGHT))
        steering_angles.append(steering_angle - STEERING_OFFSET)
    return images, steering_angles

# Get image from line and folder and column (CENTER, LEFT, RIGHT)
def getImage(line, folder, column):
    source_path = line[column]
    filename = source_path.split('/')[-1]
    current_path = './' + folder + '/IMG/' + filename
    return cv2.imread(current_path)

images = []
steering_angles = []
# Load test data from each training folder
for data in TRAINING_DATA:
    imgs, angles = loadImages(data)
    images.extend(imgs)
    steering_angles.extend(angles)

print("Generating augmented data")
augmented_images = []
augmented_steering_angles = []
# Generate augmented data - flipping the image
for image, steering_angle in zip(images, steering_angles):
    augmented_images.append(image)
    augmented_steering_angles.append(steering_angle)
    augmented_images.append(cv2.flip(image, 1))
    augmented_steering_angles.append(steering_angle*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_steering_angles)


def leNet():
    from keras.models import Sequential
    from keras.layers import Cropping2D, Lambda, Flatten, Dense, MaxPooling2D, Convolution2D

    model = Sequential()
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))
    model.add(Convolution2D(6,5,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model

model = leNet()

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=VALIDATION_PERCENTAGE, shuffle=True, nb_epoch=EPOCHS)

model.save('model.h5')
