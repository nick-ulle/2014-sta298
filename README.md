
## Cats &amp; Dogs Classification
### Christopher Aden, Chuan Qin, Nick Ulle
##### STA298, Spring 2014:
====
![alt text](https://pbs.twimg.com/profile_images/3274461853/52263042d7ca94ca26b0685d89132ba2.jpeg "GrumpyCat")
![alt text](http://cache.thephoenix.com/i/SAN/Body/versus.png "VS")
![alt text](https://pbs.twimg.com/profile_images/378800000716229938/73161235e8977a68dbeeaabc5ca303b4.jpeg "Doge")

##### Pixel-Based_Classifiers:
Principal Components Analysis: Decompose each image into a Width x Height x 3 array of RGB pixel intensities. Stack all pixel values into one vector with length 3*Width*Height. Do this for all images in the training data, then stack them into a matrix, scaling to make them all have the same width and height. Attempt to create a linear combination of the pixel values for dogs and cats, then use a linear discriminant to determine class.
*(Classification Rate: <55%)*
	
Support Vector Machines: Do a similar task as the linear discrimination in PCA, but allow for non-linear discrimination of the pixel values.
*(Classification Rate: 57%)*

##### K-means
Use spherical K-means on scaled and ZCA-whitened chunks (e.g. 16x16) of the training images, in order to build up a dictionary of K features. Images can then be mapped to a vector of length K by representing each chunk with its closest entry in the dictionary, and summing over the keys for all chunks. Classify based on the chunks.
*(Classification Rate: %)*

##### Overfeat Feature Label Classifier: 
The Overfeat program will, by default, output ImageNet categories. This classifier takes the ImageNet corpus for dog and cat words, and deterministically classifies an image as Dog or Cat, based on whether the most common feature word is a dog or cat word. If an image contains neither dog nor cat words, the image is randomly placed in Dog or Cat with equal probability. *(Classification Rate: 88.7%)*

##### Overfeat Layer Classifiers:
Extract the neural net layers from the Overfeat feature selection algorithm, pre-trained on ImageNet. Extract a 4096x1 vector from each image and store this feature vector.
	
SVM: Use the labels and 4096-vectors to build a support vector machine with a variety of kernels on a subset of the data. Test prediction rate on the held-out data. 
*(Classification Rate: 97.5%)*
	
Neural Net: Build a back-propogated neural network with 4096-length input layer and a 1D output layer. Play around with the number of hidden neurons and train until suitable number of iterations passes. Use the trained neural net to predict testing observations. 
*(Classification Rate: 96.9%)*
