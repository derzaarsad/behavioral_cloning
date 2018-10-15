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

[image1]: ./examples/center_2018_09_21_23_41_07_863.jpg "Grayscaling"
[image2]: ./examples/center_2018_09_22_00_31_53_052.jpg "Line Avoiding Image"
[image3]: ./examples/center_2018_09_22_00_31_53_128.jpg "Line Avoiding Image"
[image4]: ./examples/center_2018_09_22_00_31_53_202.jpg "Line Avoiding Image"
[image5]: ./examples/recovery.gif "Recovery Image"
[image11]: ./examples/center_2018_10_02_09_09_15_164.jpg "Normal Image"
[image12]: ./examples/center_2018_10_02_09_09_15_164_flipped.jpg "Flipped Image"
[image13]: ./examples/center_2018_09_30_01_18_39_665.jpg "No Marking Line"
[image14]: ./examples/center_2018_09_30_01_19_06_102.jpg "No Marking Line"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* behaviour-cloning.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* [model.h5](https://drive.google.com/file/d/1j4zrkwpHIVNGlctePSilZOmXHGYK6SUZ/view?usp=sharing) containing a trained convolution neural network 
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

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24, 36, 48 and 64 (model.py lines 63-80) 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 64). 

#### 2. Attempts to reduce overfitting in the model

I used only more data to combat overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (behaviour-cloning.py line 82).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, a good curve

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

On the previous project for this Self-Driving Car Nanodegree which was the Traffic Sign Classifier, I spent most of the time experimenting with the network architecture, not only
because it was required, but also because on that project I could only get more training data easily using data augmentation. This Behavioral Cloning project is different because
it provides a simulator where we can acquire the data as much as we needed, therefore on this project I wanted to focus more on the training data acquisition to combat the overfitting
instead of experimenting much with the network. First of all, I tried to use the NVIDIA network architecture introduced on the deep learning lesson as my initial network. However,
I couldn't train the network on my laptop because of the GPU memory limitation on my NVIDIA graphic card. Therefore I tried to reduce the number of weights and biases of the network by
adding two Max Pooling layer, one before the fully connected layers and one between the 5x5 and 3x3 CNN layers. Unfortunately, this still didn't solve the memory limitation problem on my
laptop, and so I made the input image smaller by adding one more Max Pooling layer directly after the cropping 2D layer. This approach worked and then I started to do the data acquisition
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

The final model architecture (behaviour-cloning.py lines 109-126) consisted of a convolution neural network with the following layers:

My final model consisted of the following layers:

**CNN**

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   						| 
| NORMALIZATION     	|                            	                |
| Cropping 2D        	| 50 from top, 20 from bottom 	                |
| Max pooling	      	| 2x2 stride,  outputs 80x160x3 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 76x156x24 	|
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 72x152x36 	|
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 68x148x48 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 34x74x48 				|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 32x72x64 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x70x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 15x35x64 				|

**Fully connected layer**

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 33600 Flattened output of CNN   		        | 
| Fully connected		| 1164 output        							|
| RELU					|												|
| Fully connected		| 100 output        							|
| RELU					|												|
| Fully connected		| 50 output        							    |
| RELU					|												|
| Fully connected		| 10 output        							    |
| RELU					|												|
| Fully connected		| 1 output        							    |
| Mean Squared Error	|           									|

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps in the initial direction and two laps in the reverse direction on track one using center lane driving. I removed all bad

data that was caused by my bad driving in the simulator, for example if the car is out of the line on the curve. Here is an example

image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct its movement when it makes a failure.

I divided the training data for the recovery into two distinct properties based what I want the vehicle to learn:

1. Learn to avoid lines
2. Learn to align itself to the center again after avoiding the line

With this data division, I didn't have to aim for a perfect recovery by data acquisition, which simplify the data acquisition process.

This image sequence shows an example of line avoiding :

![alt text][image2]
![alt text][image3]
![alt text][image4]

This animated images shows an example of recovery to the center :

![alt text][image5]

I also collected training data for a good curve. I took a combination of using a mouse and using a keyboard and utilized each of the benefit. By using a mouse we can get a smoother curve
movement whereas by using a keyboard the movement is more sudden and sharp, therefore the keyboard is good for line avoidance.

To augment the data set, I also flipped the images and their correspondent angles thinking that this would combat overfitting. By mirroring all features on one side to the other side,
it is expected that the network learns which form does the lines have instead of only learning if some features exist on the left or on the right side of the street. For example, here
is an image that has then been flipped:

![alt text][image11]
![alt text][image12]

After the collection process, I had X number of data points. I then preprocessed this data by normalizing it, and the cropped the images 50 pixels from top and 20 pixels from the
bottom.

I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The training error did not became much lower after the second epoch,
and the performance in autonomous mode is worse with after the 5th epoch based on my experience, therefore I ended using 3 as an epoch number. I used an adam optimizer so that manually training
the learning rate wasn't necessary.

After training it for a while, I realized that my effort to manually selecting the good data worth, especially by line avoidance. However, the car still failed to stay in the line
in some special curve where the line has a special mark. Some of the examples are:

![alt text][image13]
![alt text][image14]

Therefore I collected more data on these curves. After some training and data collection processes, the car can drive itself on track one without going out of the line.

The end result can be seen in this [YouTube Video](https://www.youtube.com/watch?v=Y3x_nO7U6nQ)
