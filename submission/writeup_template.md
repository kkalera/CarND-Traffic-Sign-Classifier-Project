# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?
34799 samples
* The size of the validation set is ?
4410 samples
* The size of test set is ?
12630 samples
* The shape of a traffic sign image is ?
32x32x2
* The number of unique classes/labels in the data set is ?
43

#### 2. Include an exploratory visualization of the dataset.

Visualisation provided in the notebook itself.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

My first step was to add images to the training set. Some of the classes had as little as 180 images to train and others had more than a thousand! So I chose an arbitrary number of 650 images to be the minimum. For each train label, with less than 650 images, I took a random image from the training set, and rotated it with a random value between -15 and 15 degrees.

After adding the images, I converted the images to grayscale, because the lenet network we were basing the project upon, only accepts grayscale images. I think a different network architecture using colors would have higher performance.

After all the images were converted to grayscale, I normalized the images to values between 0 and 1.

So finally I ended up with a training set of 42660 samples instead of the provided 34799.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution     		| 1x1x1x1 stride, valid padding, outputs 28x28x6|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x4 					|
| Convolution     		| 1x1x1x1 stride, valid padding,outputs 10x10x16|
| RELU					|												|
| Max pooling	      	| 1x2x2x1 stride,  outputs 5x5x16 				|
| Flatten				| outputs 400									|
| Fully connected		| input 400, output 240 						|
| RELU					|												|
| Fully connected		| input 240, output 120							|
| RELU					|												|
| Fully connected		| input 120, output 84							|
| RELU					|												|
| Fully connected		| input 84, output 43							|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model I used an Adam optimizer, with a learning rate of 0.001. I trained for 85 epochs with a batch size of 50, but the epoch count could be reduced to 50, since the model did not improve much after that.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
max 100%
* validation set accuracy of ? 
max 96.7%
* test set accuracy of ?
94.2%

* What was the first architecture that was tried and why was it chosen?
The first architecture was the LeNet architecture that was suggested in the project

* What were some problems with the initial architecture?
The model was not getting accuracy high enough to pass the bar

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
To improve the model, I added a fully connected layer, and increased the size of the existing fully connected layers.

* Which parameters were tuned? How were they adjusted and why?
I experimented a lot with batch sizes, learning rate, and epochs. 

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
The model has a few convolution layers which has been proven to work great for image analysation. A dropout layer
would help when we have a model that's overfitting.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop       			| Beware of ice/snow							| 
| Priority road			| Beware of ice/snow							|
| No entry				| Beware of ice/snow							|
| Speed limit (20km/h)	| Wild animals crossing			 				|
| Priority road			| Wild animals crossing							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

While the first image is a stop sign, the network believes it is a sign warning for ice/snow

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .03         			| Bicycles crossing								| 
| .029     				| No vehicles									|
| .028					| Priority road									|
| .027	      			| Stop					 						|
| .026				    | Keep left      								|


I'm not going to fill in the entire grid above for all examples, because they are pretty much the same. Predictions max out at about 3% and I cannot find the reason why the validation data is above 93% and the network completely fails to recognize any of the added signs.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


