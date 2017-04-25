#**Traffic Sign Recognition** 
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

[image1]: ./writeup_images/sample_train_image.png "Sample Image and Histogram Plot"
[image2]: ./writeup_images/image1_original.jpg "Traffic Sign 1"
[image3]: ./writeup_images/image2_original.jpg "Traffic Sign 2"
[image4]: ./writeup_images/image3_original.jpg "Traffic Sign 3"
[image5]: ./writeup_images/image4_original.jpg "Traffic Sign 4"
[image6]: ./writeup_images/image5_original.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/umeshpandey007/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is **34799**
* The size of the validation set is **4410**
* The size of test set is **12630**
* The shape of a traffic sign image is **(32,32,3)**
* The number of unique classes/labels in the data set is **43**

####2. Include an exploratory visualization of the dataset.

To explore the image, I decided to plot a random image and also print the corresponding label. One such image is shown below: Also as part of data exploration, I wanted to find out how many examples per class I have, for that I plotted a histogram chart for training, validation and test set. The distributions clearly show that there are few examples for some classes which gives us a fair idea that some classes might poorly behave even after training as there are few examples. Also looks like the distribution of training, validation and test examples are the same.

![Sample Image & Histogram plots][image1] 

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I decided to keep the keep the image as a RGB image and train the model. Since the input images can have different range of pixel values, it is better to apply normalization. Also, the learning rate used in the optimizer is same for all the input features(dimensions), hence it is better to have all the values of the input pixels in all dimensions to also be in the same range which helps during the gradeint descent process. For the current training images, normalization is applied with mean 0 and equal variance by performing (pixel -128)/128 operation.

I also did not added dummy images or appended any extra images to experiment with the model when sufficient data is not present.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Droput	            | 15% inputs dropped           				    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32    				|
| Droput	            | 15% inputs dropped           				    |
| Fully connected		| input: 800 output: 200             			|
| RELU					|												|
| Droput	            | 15% inputs dropped           				    |
| Fully connected		| input: 200 output: 70             			|
| RELU					|												|
| Droput	            | 15% inputs dropped           				    |
| Fully connected		| input: 70 output: 43              			|
| Softmax				|           									|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following:

optimizer: **AdamOptimizer**
batch size:  **128**
number of epochs: **15**
learning rate: **0.002**

Also, the data was shuffled at every epochs. Experimented with the featuremaps at every layer.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* validation set accuracy of **0.948**
* test set accuracy of **0.947**

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
	Initially I tried with LeNet-5 architecture from the LeNet lab. It was chosen as it had earlier known to work with handwritten digits classification with a very high validation and test accuracy.

* What were some problems with the initial architecture?
	The initial architecture when directly used for traffic sign calssification yielded only 0.89 validation accuracy. It was evident that the original architecture was not able to characterize lot of features which might be available in traffic sign classification

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
	The architecture was adjusted by adding more featuremaps in each convolutional layer. Initially experimented with using 10 featuremaps in 1st convolutional layer and 20 featuremaps in the 2nd convolutional layer. Subsequently the featuremaps in the other layers were scaled appropriately. With these changes, the validation accuracy reached 0.93 but I wanted to get a better validation and test accuracy. For that I decided to experiment with increasing the featuremap in 1st convolutional layer to 16 and 2nd convolutional layer to 32. This helped my validation accuracy to increase beyond 0.93.

* Which parameters were tuned? How were they adjusted and why?
  Tried to increase the epochs from 10 to 20, to check if validation accuracy was decreasing. Later with few trial and errors, decided the optimal value of epochs to be around 15. Also experimented with the learning rate from 0.1-0.5, 0.001 - 0.009. Found the best learning rate to be o.002, 0.001 was also good but retained it as 0.001
  
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
 I wanted to experiment with convolution and dropout to check if dropping some inputs from the previous layer helped in preventing overfitting. With trial and experimentation, found dropout layers with dropping 15% of inputs to subsequent layers prevented in overfitting the data. Also the accuracy at different iterations turned out to be pretty stable.

If a well known architecture was chosen:
* What architecture was chosen?
	LeNet-5 was chosen as it was widely known to be successful in handwritten digits classification.
* Why did you believe it would be relevant to the traffic sign application?
	The traffic sign classification seems to be similar to handwritten digits calssification, except it has lot of features to detect, hence the architecture was believed will also work in traffic sign classification.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
	The validation set accuracy and test accuracy for the modeled distributions turns out to be pretty close which suggests the model is working well without any overfitting.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Right of way at next intersection][image2] ![Road Work][image3] ![Speed 60][image4] 
![Stop Sign][image5] ![Roundabout][image6]

Image like 60km/hr sign when rescaled can have problems as it appears to be taken at a different angle as compared to other images which don't seem to have any angular distortions. The other features like presence of other external factors like trees, railings, background as in case of Stop sign might make it difficult to classify. After resizing the images, without maintaining the aspect ratio, might result in some extraneous features to be falsely used in training amd also impact the training process.

I tried cv2 package to scale the images in the code, but seems for RGB image, the cv2.resize function did not work well. Hence to resize the image to 32x32x3, I used an external image editor to resixe the image appropriately.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right of way at next intersection      		| Right of way at next intersection| 
| Roundabout mandatory     			| Roundabout mandatory 										|
| Stop					| Priority Road											|
| 60 km/h	      		| Priority Road					 				|
| Road Work			| Road Work     							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 14th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a **right of way at next intersection** (probability of 1.0), and the image does contain a **right of way ay next intersection** sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| right of way at next intersection   									| 
| 3.1e-18      				| Roundabout mandatory										|
| 4.4e-19					| 100 km/h											|
| 2.738e-19	      			| Ice/Snow					 				|
| 2.483e-20				    | General caution      							|


For the second image, the model is relatively sure that this is a **Roundabout mandatory** (probability of 1.0), and the image does contain a **Roundabout mandatory** sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Roundabout mandatory   									| 
| 3.02e-12      				| Keep right										|
| 2.74e-13					| Priority road											|
| 7.79e-15	      			| Ice/snow					 				|
| 1.83e-15				    | Dangerous curve-right      							|

For the third image, the model fails to correctly recognize the image as **stop sign**. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9         			| Priority Road   									| 
| 8.69e-8      				| Yield										|
| 2.10e-8					| Keep Right											|
| 7.59e-11	      			| End of no passing by vehicles over 3.5 metric tons					 				|
| 4.84e-11				    | No passing   							|

For the thfourthird image, the model fails to correctly recognize the image as **60 km/hr**. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9         			| Priority Road   									| 
| 8.69e-4      				| Go straight or left	|
| 4.94e-5					| Ahead only											|
| 4.29e-5	      			| Keep Left					 				|
| 3.87e-5				    | right of way at next intesection      							|


For the last image, the model is relatively sure that this is a **Road Work** (probability of 0.9), and the image does contain a **Road Work** sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9         			| Road work   									| 
| 2.98e-6      				| Ice/Snow										|
| 1.36e-8					| General caution											|
| 5.19e-9	      			| Road narrows on the right					 				|
| 2.08e-9				    | 80km/hr     							|


###Final Thoughts
I would like to experiment further with following things in mind, since the example images from web which failed miserably were not among the top 5 probabilities:
1. Input the model with grayscale images
2. Increase the number of training set examples so that the distribution is as much as uniform for all the classes.
3. Experiment with increasing the convolutional layers instead of just using the LeNet-5 architecture. Maybe around 7.
4. Understand the output image of each layer 1 and get intution of the featuremaps.
5. Add some random noise into training set and check the behaviour of network


























































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































 





































