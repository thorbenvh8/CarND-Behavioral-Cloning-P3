# **Behavioral Cloning**
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is a normal LeNet implementation.
The data is normalized and cropped (top: 50px, bottom: 20px).

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 56-59).


The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 94).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to learn by Behavioral Cloning.

My first step was to use the simpliest neural network model to get used to the whole process of generating the model and testing it via the car app using autonomous driving.

The model didnt work that well, the car was barely moving forward, but trying to move a lot left and right.

My second step was to use a convolution neural network model similar to the LeNet. I thought this model might be appropriate because it is about making decisions on images.
As well I added augmented images that where just a horizontal flip of the already existing data and inverting the steering angle.

The model was working better and was overfitting already after 4 epochs. The problem was curves and getting back to center of the road.

The next step was to use not only the center image, but as well the left and right image and add a correction of 0.2 to make it move into the center.

The model was overfitting after 7 epochs. The model worked pretty good, even in curves, but sometimes got totally lost and just drove off the road.

The final step was to crop the image from the top 50pxs and from the bottom 20px.

The loss went down dramatically but it took 10 epochs after it was overfitting.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network based on LeNet.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior I created the following training sets:

- training1
 - driving one round
- training2
 - driving another round
- training3
 - driving one round reverse
- training4
 - driving one round little moving left and ride from the center
- training5
 - single recordings of moving the car into center again

To augment the data sat, I also flipped images and angles thinking that this would even up left and right corners in the training data.

After the collection process, I had 35046 number of data points. I then preprocessed this data by normalizing and cropping (for more information see above).

I finally randomly shuffled the data set and put 20% of the data into a validation set (28036 training data points, 7010 validation data points).

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10. Throughout the development it moved down to 4 and raised up to 10 again. I used an adam optimizer so that manually training the learning rate wasn't necessary.
