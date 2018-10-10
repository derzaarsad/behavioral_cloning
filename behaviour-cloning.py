import os
import csv
import cv2
import numpy as np
from random import shuffle
import sklearn

def add_sample_from_file(filename,samples):
	with open('../driving_log_' + postfix + '.csv') as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			if(len(line) > 1):
				line[0] = line[0].replace("IMG","IMG_" + postfix)
				samples.append(line)
	return samples

samples = []
samples = add_sample_from_file('../driving_log_210918.csv',samples)
samples = add_sample_from_file('../driving_log_curvy1.csv',samples)
samples = add_sample_from_file('../driving_log_curvy2.csv',samples)
samples = add_sample_from_file('../driving_log_recovery1.csv',samples)
samples = add_sample_from_file('../driving_log_300918.csv',samples)
samples = add_sample_from_file('../driving_log_curve021018.csv',samples)
samples = add_sample_from_file('../driving_log_curve031018.csv',samples)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]
			
			images = []
			angles = []
			for batch_sample in batch_samples:
				name = batch_sample[0]
				center_image = cv2.imread(name)
				center_angle = float(batch_sample[3])
				images.append(center_image)
				angles.append(center_angle)
				image_flipped = np.fliplr(center_image)
				measurement_flipped = -center_angle
				images.append(image_flipped)
				angles.append(measurement_flipped)
			
			# trim image to only see section with road
			X_train = np.array(images)
			y_train = np.array(angles)
			yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(MaxPooling2D())
model.add(Convolution2D(24,(5,5),activation='relu',padding='valid'))
model.add(Convolution2D(36,(5,5),activation='relu',padding='valid'))
model.add(Convolution2D(48,(5,5),activation='relu',padding='valid'))
model.add(MaxPooling2D())
model.add(Convolution2D(64,(3,3),activation='relu',padding='valid'))
model.add(Convolution2D(64,(3,3),activation='relu',padding='valid'))
model.add(MaxPooling2D())
#model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

model.save('model.h5')