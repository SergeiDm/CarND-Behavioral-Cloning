# CarND-Behavioral-Cloning
## Project Description
The project consists of learning network to map images taken from a front-facing camera with steering angles. According to NVIDEA [«End to End Learning for Self-Driving Cars»](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf), this approach for self-driving cars may lead to better performance in comparison explicit decomposition of the self-driving problem.

## Project files
The project includes the following folder/files:
- model.py – the script to create and train the model.
- drive.py – for driving the car in autonomous mode.
- model.h5 – the model weights.
- model.json – the model architecture.
- illustrations - the folder with pictures 'for README.md'.

Result video of this project:

<a href="https://www.youtube.com/watch?v=wug6ksRY5BQ" target="_blank"><img src="http://img.youtube.com/vi/wug6ksRY5BQ/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a>

For testing autonomous driving the following code should be executed:

`python drive.py model.json`

## Project Data
Input data is a set of RGB images 160 (height) x 320 (width) pixels, generated by the driving simulator in training mode. 

Examples:

![Image examples](https://github.com/SergeiDm/CarND-Behavioral-Cloning/blob/master/illustrations/image_examples.jpg)

The total number of images used in this project is 14 821. ~30% of them is recovery data.

Output data is steering angles in range [-1, 1].

## Project Pipeline
### Collecting data
All input and output data were recorded by simulator in training mode.
### Preprocessing data
Output data is in range [-1, 1], so there is no need for transformation like normalization.

For input data, the following steps were applied:
- Cropping images for getting value information. An image consists of a lot of irrelevant details (e.g. sky, trees, lake), which prevent network to produce value results. Important things here are lane lines and driving surface. From initial images, 60 pixels from the top and 20 pixels from the bottom were removed, so new image size is 80 x 320 pixels.
- Color converting. There are few places in the road where it’s complicated to distinguish road surface and landscape. It leads the model to produce incorrect results. In order to make elements more distinguishable HSV color space was applied to the images.

![Preprocessing data](https://github.com/SergeiDm/CarND-Behavioral-Cloning/blob/master/illustrations/preprocessing_data.jpg)

In Fig. 4 the right border is more recognizable.
- Normalizing input data by min-max transformation from range [0, 255] to [0, 1]. This technique allows keeping weights small.

### Training and validating model
Data generated by simulator was used as input for convolutional neural network (see CNN architecture and Training strategy).

### Testing model
Model was tested on track 1 of the simulator. The main condition is the vehicle should be in the road and no tire leaves drivable part.

## CNN architecture
Architecture of network depends on input and output data (count, quality, how they were preprocessed). Initially, I used the architecture proposed by NVIDEA for self-driving cars problem, but there were two places where the vehicle tires intersected lane lines, so some changes were applied. 

The final structure of the model is:

![Net architecture](https://github.com/SergeiDm/CarND-Behavioral-Cloning/blob/master/illustrations/net_architecture.jpg)

The model have the following differences from original NVIDEA’s one:
- Inception model between layer 1 and 3. As a rule, inception model gives better performance and used in well-known networks (e.g. GoogLeNet) for image classification. In this case, inception model concatenates outputs from 5x5 conv layer and 2x2 max pooling of 1x1 conv layer.
- Numbers of neurons in layers were insignificantly increased during validating and testing model.

## Training strategy
All collected data was divided into training (90%) and validating (10%) sets by using random splitting (train_test_split). Comparing training and validating loss helped to control if the model was over or under fitting.
In order to prevent training stage from out of memory, generator was used. Generator produces a batch of data and gives it to the model for learning process.
Other features of training stage:
- Objective/ loss function – mean square error.
- Type of optimizer: Adam optimizer. This gradient-based optimizer changes learning rate and takes advantage of SGD with momentum.
- Type of activation function: tanh. This function not only introduce non-linearity, but also its output [-1, 1] coincides with outputs of the model:

![tanh_function](https://github.com/SergeiDm/CarND-Behavioral-Cloning/blob/master/illustrations/tanh_function.jpg)

- Batch size – 128 images. Good batch size increases convergency and the same time keeps computational cost low. This hyperparameter was chosen after several starts of training.
- Epochs - 2. More epochs didn’t significantly improve training and validating loss. Moreover, large number of epochs may lead to overfitting.
- Dropout – 20%. In order to prevent model from overfitting dropout techniques was applied. Common range for this hyperparameter is 20-50%.

By using 90% of 14 821 images and two epochs, the total number of images processed by network is 2 * 0.9 * 14821 = 26 677. For computers, which don’t use GPU-accelerated learning, processing this number of images doesn’t take much time.

## Testing model
Model was tested on the simulator track, from which pictures for learning were taken. A vehicle keep moving on drivable part of the road.
