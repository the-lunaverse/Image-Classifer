# Image-Classifer

Data Source: 
The CIFAR-10 dataset:https://www.cs.toronto.edu/~kriz/cifar.html

### The analysis demonstrates the process of creating, training, and evaluating a Convolutional Neural Network (CNN) model in Python for image classification with a focus on ship detection by:

-Filtering out ship images (label 8) from both training and test data.

-Combining ship and non-ship images for training and test data.

-Preprocessing the data by normalizing image pixel values and converting labels to categorical format.

-Defining the CNN model architecture with three Conv2D and MaxPooling2D layers, followed by a Flatten layer, a Dense layer with dropout, and the output layer with softmax activation.

-Compiling the model with the Adam optimizer and a learning rate of 0.001.

-Training the model for 20 epochs with a batch size of 64 and 20% validation split.

-Evaluating the model on test data and displaying the test accuracy.

-Visualizing the model's training and validation loss and accuracy over the 20 epochs using line plots.

-Creating a confusion matrix of the model's predictions and visualizing it as a heatmap to display the performance in classifying ship and non-ship images.
