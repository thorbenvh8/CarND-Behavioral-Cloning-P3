import csv
import cv2
import numpy as np

VALIDATION_PERCENTAGE=0.2
EPOCHS=6

TRAINING_DATA = ['training1', 'training2', 'training3', 'training4', 'training5']

IMG_CENTER = 0
IMG_LEFT = 1
IMG_RIGHT = 2
STEERING_ANGLE = 3
THROTTLE = 4
BREAK = 5
SPEED = 6

def loadImages(folder):
    lines = []
    with open('./' + folder + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    images = []
    steering_angles = []
    for line in lines:
        source_path = line[IMG_CENTER]
        filename = source_path.split('/')[-1]
        current_path = './' + folder + '/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        steering_angle = float(line[STEERING_ANGLE])
        steering_angles.append(steering_angle)
    return images, steering_angles

images = []
steering_angles = []
for data in TRAINING_DATA:
    imgs, angles = loadImages(data)
    images.extend(imgs)
    steering_angles.extend(angles)

X_train = np.array(images)
y_train = np.array(steering_angles)


def leNet():
    from keras.models import Sequential
    from keras.layers import Lambda, Flatten, Dense, MaxPooling2D, Convolution2D

    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
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
