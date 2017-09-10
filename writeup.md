#**Behavioral Cloning**

## Prequiries

install missing packages not part of anaconda distribution

    pip install python-socketio
    pip install eventlet

### Potential errors

- poor predictions on training and validating -> underfitting -> more layers, more epochs
- poor on validation -> overfitting -> dropout, fewer convolutions, fewer fully connected, collect more data & augment

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on the [NVIDIA net](https://arxiv.org/pdf/1604.07316v1.pdf)
which was developed specifically for autonomous driving tasks. The original
NN looked like follows:

![test](./writeup/NVIDIA-net.png)

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. Also, images are cropped: 70px at the top
of the images (unnecessary sky, trees, etc.) and 25px at the bottom (hood of
the car) are not used for training, since carrying no helpful (or even
distracting) information.

#### 2. Attempts to reduce overfitting in the model

To prevent overfitting like displayed in the following image, I implemented
**dropout** after every Convolution and fully-connected network layer ![test](./writeup/01_performance.png)

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer that implements **weight decay**, so the learning rate was not tuned manually. An initial `LEARNING_RATE = 0.001` yielded best results.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of

- a different person drove 2 laps on track 1 forward.
- a different person drove 2 laps on track 1 backwards.
- a different person drove 2 laps on track 2 forward.
- a different person drove 2 laps on track 2 backwards.
- I'm driving the crital part (exit to dust road couple of times in both directions).
- I'm driving in the center for 2 laps forward.
- I'm driving in the center for 2 laps backwards.
- I'm driving 1 laps, recording is only turned on when recovering from dirt back to
 the road.
- I'm driving around curves when red/white road marking is present.

Each stored in a separate sub-folder to allow to use all, or only a subset of
training patterns. For training, I used all of them. For each training sample,
I added a flipped version with inverted steering angle to the training set as well.
This serves two purposes: Balancing the dataset between left and right steering
actions and generating more training data.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I implemented
**dropout** after every Convolution and fully-connected network layer.

Then I implemented **early stopping** to stop training when the performance converges (and neither earlier wasting model performance nor later wasting time). This is configuration `EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3, verbose=0, mode='auto')` I used.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I recorded myself passing them a few times. Examples were
the exit to the dirt road and some places where the street was marked by white/red
areas.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture was a modification of the [NVIDIA net](https://arxiv.org/pdf/1604.07316v1.pdf). Instead of having 4 fully-connected
layers with 1164/100/50 and 10 neurons at the end, I ended up using 120/32 and 1
neuron. This combination was the result of a long trial-and-error periode.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text](./writeup/NVIDIA-net_mod.png)

Here's how the training went

    79s - loss: 0.0581 - val_loss: 0.0550
    Epoch 2/50
    76s - loss: 0.0511 - val_loss: 0.0460
    Epoch 3/50
    75s - loss: 0.0479 - val_loss: 0.0462
    Epoch 4/50
    76s - loss: 0.0448 - val_loss: 0.0432
    Epoch 5/50
    76s - loss: 0.0422 - val_loss: 0.0382
    Epoch 6/50
    75s - loss: 0.0406 - val_loss: 0.0451
    Epoch 7/50
    74s - loss: 0.0401 - val_loss: 0.0415
    Epoch 8/50
    75s - loss: 0.0385 - val_loss: 0.0426
    Epoch 9/50
    74s - loss: 0.0388 - val_loss: 0.0331
    Epoch 10/50
    74s - loss: 0.0374 - val_loss: 0.0389
    Epoch 11/50
    75s - loss: 0.0372 - val_loss: 0.0382
    Epoch 12/50
    73s - loss: 0.0354 - val_loss: 0.0315
    Epoch 13/50
    74s - loss: 0.0349 - val_loss: 0.0350
    Epoch 14/50
    73s - loss: 0.0351 - val_loss: 0.0316
    Epoch 15/50
    75s - loss: 0.0344 - val_loss: 0.0286
    Epoch 16/50
    73s - loss: 0.0338 - val_loss: 0.0313
    Epoch 17/50
    73s - loss: 0.0333 - val_loss: 0.0305
    Epoch 18/50
    73s - loss: 0.0334 - val_loss: 0.0287
    Epoch 19/50
    73s - loss: 0.0329 - val_loss: 0.0316
    early stopping: stopped.

The graph looks like this

![alt text](./writeup/14_performance.png)


#### 3. Creation of the Training Set & Training Process

After the collection (as described aboce) process finished, I had roughly 6500
 number of data points.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I set the number of epochs to 50 and implemented
early stopping as mentioned above. I used an adam optimizer so that manually training the learning rate wasn't necessary.

### Info for myself: failures

- 03 to 04: Reduced Dense from 120/50/10/1 to 120/64/1. Much less overfitting.
- 05: Reduced correction from 0.1 to 0.05
- 06: Removed "bernhard_forward_center2", "bernhard_reverse_center2"
- 07: Removed "bernhard_critical_part", increased ``EPOCHS`` from 5 to 10
- 08: ``EPOCHS`` back to 5, "annika_reverse", "annika_forward", "bernhard_critical_part",
"bernhard_forward_center2", "bernhard_reverse_center2",
"bernhard_forward_recovery"
- 09: From ``lr=0.0001`` ``to lr=0.001``, added "bernhard_red", "annika_2_reverse", "annika_2_forward
- 11: Removed 2nd ``Conv2D(64, (3, 3))``
- 12: Removed 2 Dense layer from NVIDIA net (2 remaining (64/1))
