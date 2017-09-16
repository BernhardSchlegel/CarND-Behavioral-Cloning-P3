import cv2
import csv
import numpy as np
import os
from sklearn.utils import shuffle
import sklearn
import matplotlib.pyplot as plt

import common

# Settings
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0001
REGULARIZATION = 0.001
STEERING_CORRECTION = 0.25  # this is a parameter to tune

lines = []
for chosen_folder in ["udacity",
                      "bernhard_soft_recovery"]:
    with open("./data/" + chosen_folder + "/driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)

        for line in reader:
            lines.append((chosen_folder, line))

print("loading and enhancing images.")

def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def generator(samples, batch_size=BATCH_SIZE):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for chosen_folder, line in batch_samples:
                for i in range(3):
                    source_path = line[i]
                    filename = os.path.split(source_path)[-1]  # source_path.split("/")[-1]
                    current_path = "./data/" + chosen_folder + "/IMG/" + filename
                    image = cv2.imread(current_path)

                    # convert from RGB to YUV
                    image = common.preprocess_image(image, mode="train")

                    measurement = float(line[3])
                    if i is 0:
                        # center camera
                        measurement_corrected = measurement
                    elif i is 1:
                        # left cam
                        measurement_corrected = measurement + STEERING_CORRECTION
                    elif i is 2:
                        # right cam
                        measurement_corrected = measurement - STEERING_CORRECTION

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

# balancing the distribution
def plot_hist_get_angles(lines, num_bins=21):

    angles = []
    for elem in lines:
        angles.append(float(elem[1][3]))
    plt.hist(angles, bins=num_bins)  # make sure that it's a odd number
    plt.show(block=False)
    plt.savefig('norm.png')

    return angles

num_bins = 25
angles = plot_hist_get_angles(lines, num_bins=num_bins)
target_elements_per_bin = len(angles) / num_bins
hist, bins = np.histogram(angles, np.dot(list(range(-12, 13)), 0.1))
# calculate the probablity to be kept for each bin
keep_probalities = np.dot(target_elements_per_bin, 1 / hist)

# remove according to probability
keep_list = []
for i in range(len(angles)):
    keep = True
    for j in range(num_bins):
        if angles[i] > bins[j] and angles[i] <= bins[j+1]:
            # delete from X and y with probability 1 - keep_probs[j]
            if np.random.rand() > keep_probalities[j]:
                keep = False
                break
    keep_list.append(keep)

from itertools import compress
lines = list(compress(lines, keep_list))
plot_hist_get_angles(lines)

# split samples
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

print("importing keras")
from keras import optimizers
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Activation, Dropout, Cropping2D, ELU
from keras.layers.pooling import MaxPooling2D
from keras import optimizers # to enable custom optimizer
from keras import regularizers

print("building model")
model = None

architecture = "NVIDIA_mod"

IMG_DIM_Y = 160 #66       # 160
IMG_DIM_X = 320 #200      # 320
IMG_CROP_Y_TOP = 70 #38  # 70
IMG_CROP_Y_BOT = 25 #10  # 25



if architecture == "NVIDIA":
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
elif architecture == "NVIDIA_mod":
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(IMG_DIM_Y, IMG_DIM_X, 3)))
    model.add(Cropping2D(cropping=((IMG_CROP_Y_TOP, IMG_CROP_Y_BOT), (0, 0))))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="elu", kernel_regularizer=regularizers.l2(REGULARIZATION)))
    #model.add(Dropout(0.5))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="elu", kernel_regularizer=regularizers.l2(REGULARIZATION)))
    #model.add(Dropout(0.5))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="elu", kernel_regularizer=regularizers.l2(REGULARIZATION)))
    #model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3), activation="elu", kernel_regularizer=regularizers.l2(REGULARIZATION)))
    #model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3), activation="elu", kernel_regularizer=regularizers.l2(REGULARIZATION)))
    #model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(120))
    #model.add(Dropout(0.5))
    model.add(Dense(50))
    #model.add(Dropout(0.5))
    model.add(Dense(10))
    #model.add(Dropout(0.5))
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
my_adam = optimizers.Adam(lr=LEARNING_RATE)
model.compile(loss='mse', optimizer=my_adam)

from keras.callbacks import EarlyStopping
my_callbacks = [
    EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=0, mode='auto')
]

history_object = model.fit_generator(train_generator,
                                     steps_per_epoch=len(train_samples)/BATCH_SIZE,
                                     validation_data=validation_generator,
                                     validation_steps=len(validation_samples)/BATCH_SIZE ,
                                     epochs=EPOCHS, verbose=2, callbacks=my_callbacks)
### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.figure(2)
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
plt.savefig('training.png')


model.save("model.h5")
print("model saved.")
exit()

