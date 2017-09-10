import cv2
import csv
import numpy as np
import os

chosen_folder = "annika_reverse"
lines = []
with open("./data/" + chosen_folder + "/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)

    for line in reader:
        lines.append(line)

images = []
measurements = []

print("loading and enhancing images.")
first = True
for line in lines:
    # cam_center,cam_left,cam_right,steering,throttle,brake,speed
    if first:
        # skip first cause of old udacity log format
        first = False
        continue

    correction = 0.1  # this is a parameter to tune
    for i in range(3):
        source_path = line[i]
        filename = os.path.split(source_path)[-1]  # source_path.split("/")[-1]
        current_path = "./data/" + chosen_folder + "/IMG/" + filename
        image = cv2.imread(current_path)

        measurement = float(line[3])
        if i is 0:
            # center camera
            measurement_corrected = measurement
        elif i is 1:
            # left cam
            measurement_corrected = measurement + correction
        elif i is 2:
            # right cam
            measurement_corrected = measurement - correction

        # append original
        images.append(image)
        measurements.append(measurement_corrected)

        # flip n append
        image_flipped = np.fliplr(image)
        measurement_flipped = -measurement_corrected
        images.append(image)
        measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

print("importing keras")
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Activation, Dropout, Cropping2D
from keras.layers.pooling import MaxPooling2D

print("building model")
model = None

architecture = "NVIDIA"

if architecture == "NVIDIA":
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Conv2D(24, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Conv2D(36, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Conv2D(48, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Conv2D(64, 3, 3, activation="relu"))
    model.add(Conv2D(64, 3, 3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

elif architecture == "LeNET":
    model = Sequential()

    model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Conv2D(6, 5, 5, border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation("sigmoid"))

    model.add(Conv2D(16, 5, 5, border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation("sigmoid"))
    model.add(Dropout(0.5))

    model.add(Conv2D(120, 1, 1, border_mode='valid'))

    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))


print("starting training")
# TODO: Use RMSE as metric.
model.compile(loss="mse", optimizer="adam")
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3,
          verbose=1)  # verbose 1 outputs progress bar and loss in the terminal.

model.save("model.h5")
print("model saved.")
exit()

