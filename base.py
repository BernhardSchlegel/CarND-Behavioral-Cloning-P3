import cv2
import csv
import numpy as np
import os
from sklearn.utils import shuffle
import sklearn

# Settings
BATCH_SIZE = 32
EPOCHS = 5

lines = []
for chosen_folder in ["annika_reverse", "annika_forward"]:
    with open("./data/" + chosen_folder + "/driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)

        for line in reader:
            lines.append((chosen_folder, line))

print("loading and enhancing images.")
def generator(samples, batch_size=BATCH_SIZE):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for chosen_folder, line in batch_samples:
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
                    angles.append(measurement_corrected)

                    # flip n append
                    image_flipped = np.fliplr(image)
                    measurement_flipped = -measurement_corrected
                    images.append(image)
                    angles.append(measurement)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# split samples
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

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
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
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
import matplotlib.pyplot as plt
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator,
                                     steps_per_epoch=len(train_samples),
                                     validation_data=validation_generator,
                                     validation_steps=len(validation_samples),
                                     epochs=5, verbose=1)
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


model.save("model.h5")
print("model saved.")
exit()

