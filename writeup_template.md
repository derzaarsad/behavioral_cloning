# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

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

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

On the previous project for this Self-Driving Car Nanodegree which was the Traffic Sign Classifier, I spent most of the time experimenting with the network architecture, not only

because it was required, but also because on that project I could only get more training data easily through data augmentation. This Behavioral Cloning project is different because

it provides a simulator where we can acquire the data as much as we needed, therefore on this project I wanted to focus more on the training data acquisition to combat the overfitting

instead of experimenting much with the network. First of all, I tried to use the NVIDIA network architecture introduced on the deep learning lesson as my initial network. However,

I couldn't train the network on my laptop because of the GPU memory limitation on my NVIDIA graphic card. Therefore I tried to reduce the number of weights and biases of the network by

adding two Max Pooling layer, one before the fully connected layers and one between the 5x5 and 3x3 CNN layers. Unfortunately, this still didn't solve the memory limitation problem on my

laptop, and so I made the input image smaller by adding one more Max Pooling layer directly after the normalization. This approach worked and then I started to do the data acquisition

using the simulator.

I spent most of the time selecting the data for training, and also because I was working with so many data, the training time was much longer than the training time for the Traffic Sign

Classifier, therefore it was also not easy to only use more data to combat overfitting. At one time, I felt that my network suffered from overfitting because the validation set error

stayed in the 1e-2 region while the training set error slowly became much smaller, and so I used dropout with the intention to regularize the network, however, the network performed much

worse, therefore I used an early breakup instead of dropout so that the network is not too optimized for the training data. I didn't change the architecture quite much from the initial design,

because I saw that the network performed quite well with a "dirty" training data that I acquired by myself. And also the training time was very long that I could not easily make a small change

to fine tune the network because of the time limitation that I had. After I reached a level where the network works quite well with the mentioned architecture and an early breakup, I iteratively

approached the solution by training, analyzing the result, adding more data and then trained the network again until the simulated car can drive very well. At the end of the process, the vehicle

is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
