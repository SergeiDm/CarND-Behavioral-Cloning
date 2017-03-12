# P3 Behavioral Cloning

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from os import path, makedirs

# directory with image files and logs
PATH = 'A:\\data18hsv_full\\'
# constant for cropping image from top and bottom
H_START = 60
H_END = 140

# reading logs, transforming images and creating a new log file, contained only paths and steering angles
driving_log = pd.DataFrame()
# checking whether new log file was created or not
if not(path.isfile(PATH+'driving_log2.csv')): 
    driving_log = pd.read_csv(PATH+'driving_log.csv', 
                              header=None,
                              names=['Image_name', 'Steering_angle'],
                              usecols=[0, 3])
    h, w, d = mpimg.imread(driving_log.loc[0,'Image_name']).shape
    j = 1
    k = 1
    # creating new folders for transformed images
    makedirs(PATH+'preproc_data'+str(j)+r'\\')  
    for i in range(len(driving_log)):
        if ((i % 5000)==0) and (i!=0):
            j +=1
            makedirs(PATH+'preproc_data'+str(j)+r'\\')
            k = 1
        x = mpimg.imread(driving_log.loc[i,'Image_name'])[H_START:H_END, :]
        x = cv2.cvtColor(x, cv2.COLOR_RGB2HSV_FULL)
        plt.imsave(PATH+'preproc_data'+str(j)+r'\\'+'img_'+str(k)+'.jpg', x)
        driving_log.loc[i,'Image_name'] = path.dirname(PATH)+r'\preproc_data'+str(j)+r'\img_'+str(k)+'.jpg'
        k += 1       
    driving_log.to_csv(PATH+'driving_log2.csv', header=['Image_name', 'Steering_angle'], index=False)
    h -= h + H_START - H_END
else:
    driving_log = pd.read_csv(PATH+'driving_log2.csv', header=0)
    h, w, d = mpimg.imread(driving_log.loc[0,'Image_name']).shape

# importing function for dividing data into train and validation sets 
from sklearn.cross_validation import train_test_split

# spliting input data into train (90%) and validation (10%) sets
driving_log_train, driving_log_valid = train_test_split(driving_log,
                                                       test_size=0.01)
                                                              
del driving_log

# creating new indices
driving_log_train = driving_log_train.reset_index(drop=True)
driving_log_valid = driving_log_valid.reset_index(drop=True)

# function for normalizing data
def norm(X):
    min_old = 0
    max_old = 255    
    min_new = 0
    max_new = 1
    return (X-min_old)*(max_new-min_new)/(max_old-min_old)+min_new

# creating model
# constants
# probability of dropping neurons
P_DROP = 0.2
# number of epochs
EPOCHS = 2
# number of images after them back propagation will be used
BATCH_SIZE = 128
# activation function
ACTIV = 'tanh'
# weights constraint
C = 4

from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, Input, merge
from keras.layers.convolutional import Convolution2D
from keras.constraints import maxnorm
from keras.layers.pooling import MaxPooling2D 

# input size 80x320x3
inputs = Input(shape=(h, w, d,))

# convolutional layer 1: filter-5x5, strides-2x2, valid padding, output-38x158x24
conv2d_1 = Convolution2D(24, 5, 5,
                        activation=ACTIV,
                        border_mode='valid',
                        subsample=(2,2),
                        W_constraint=maxnorm(C))(inputs)

# inception model for convolutional layer 2
# convolutional layer 2_1: filter-1x1, strides-1x1, same padding, output-38x158x32
conv2d_2_1 = Convolution2D(32, 1, 1,
                        activation=ACTIV,
                        border_mode='same',
                        subsample=(1,1),
                        W_constraint=maxnorm(C))(conv2d_1)
# maxpooling: pool-2x2, strides-2x2, same padding, output-19x79x32
max_pool_2_1 = MaxPooling2D(pool_size=(2,2),
                          strides=(2,2),
                          border_mode='same')(conv2d_2_1)

# convolutional layer 2_2: filter-5x5, strides-2x2, same padding, output-19x79x32
conv2d_2_2 = Convolution2D(32, 5, 5,
                        activation=ACTIV,
                        border_mode='same',
                        subsample=(2,2),
                        W_constraint=maxnorm(C))(conv2d_1)
# merging max_pool_2_1 and convolutional layer 2_2, output 19x79x64
conv2d_2 = merge([max_pool_2_1, conv2d_2_2], mode='concat', concat_axis=3)

# convolutional layer 3: filter-5x5, strides-2x2, valid padding, output-8x38x64
conv2d_3 = Convolution2D(64, 5, 5,
                        activation=ACTIV,
                        border_mode='valid',
                        subsample=(2,2),
                        W_constraint=maxnorm(C))(conv2d_2)

# convolutional layer 4: filter-3x3, strides-2x2, valid padding, output-3x18x72
conv2d_4 = Convolution2D(72, 3, 3,
                        activation=ACTIV,
                        border_mode='valid',
                        subsample=(2,2),
                        W_constraint=maxnorm(C))(conv2d_3)

# convolutional layer 5: filter-3x3, strides-2x2, valid padding, output-1x16x72
conv2d_5 = Convolution2D(72, 3, 3,
                        activation=ACTIV,
                        border_mode='valid',
                        subsample=(1,1),
                        W_constraint=maxnorm(C))(conv2d_4)
# flattening 
flatten_conv2d_5 = Flatten()(conv2d_5)

# fully-connected layer1: 512 neurons
fc1 = Dense(512, activation=ACTIV, W_constraint=maxnorm(C))(flatten_conv2d_5)
fc1 = Dropout(P_DROP)(fc1)

# fully-connected layer2: 128 neurons
fc2 = Dense(128, activation=ACTIV, W_constraint=maxnorm(C))(fc1)
fc2 = Dropout(P_DROP)(fc2)

# fully-connected layer3: 24 neurons
fc3 = Dense(24, activation=ACTIV, W_constraint=maxnorm(C))(fc2)
fc3 = Dropout(P_DROP)(fc3)

# output layer
outputs = Dense(1, activation=ACTIV)(fc3)

# creating Keras model 
model = Model(input=inputs, output=outputs)

# configurating the learning process: Adam optimizer, objective - mean squared error
model.compile(loss='mean_squared_error',
              optimizer='adam')

# defining generator
length = len(driving_log_train)
length1 = len(driving_log_valid) 

def generator_train():
    global driving_log_train, gh
    while 1:
        for i in range(0, length, BATCH_SIZE):
            X_data_train = []
            y_data_train = []
            k = min(i+BATCH_SIZE, length)
            # reading training images
            for j in range(i, k):
                X_data_train.append(mpimg.imread(driving_log_train.loc[j,'Image_name']))
                y_data_train.append(driving_log_train.loc[j,'Steering_angle'])
            X_data_train = norm(np.array(X_data_train))
            y_data_train = np.array(y_data_train)
            yield X_data_train, y_data_train
        # shuffle data before new epoch
        index = driving_log_train.index.tolist()
        np.random.shuffle(index)
        driving_log_train = driving_log_train.ix[index]
        driving_log_train = driving_log_train.reset_index(drop=True)

# reading data for valid set         
X_data_valid = []
y_data_valid = []
for i in range(0, length1):
    X_data_valid.append(mpimg.imread(driving_log_valid.loc[i,'Image_name']))
    y_data_valid.append(driving_log_valid.loc[i,'Steering_angle'])
X_data_valid = norm(np.array(X_data_valid))
y_data_valid = np.array(y_data_valid)
        
# training model
history = model.fit_generator(generator=generator_train(),
                              samples_per_epoch=length,
                              nb_epoch=EPOCHS,
                              verbose=1,
                              validation_data=(X_data_valid, y_data_valid))

# save the model and weights
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('model.h5')

del X_data_valid, y_data_valid


