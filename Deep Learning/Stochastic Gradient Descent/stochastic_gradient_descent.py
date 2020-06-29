# -*- coding: utf-8 -*-
"""
Created on Fri May 15 17:19:39 2020

@author: Rahul Sapireddy
"""

# import the requiredpackages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Learning Rate -> The size of steps covered during each iteration (alpha)
# Epoch -> When the entire dataset is passed to the model
# define Learning rate and epochs
learning_rate = 0.001
no_of_epochs = 100
batch_size = 32

# generate a sample data se with two feature columns and one class column
# features with random numbers and class with either 0 or 1 
# total number of samples will be 1000
(X,y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
# print(X)
# print(y)

print(y.shape)
# convert y into double indexed form
y = y.reshape((y.shape[0],1))
print(y.shape)

# reshape X value to include a 1's column to accomodate the weight column due to the bias trick
X = np.c_[X, np.ones((X.shape[0]))]
# print(X)

# split 50% (500 rows) for training and rest 50% for testing
(trainX, testX, trainY, testY) = train_test_split(X,y, test_size = 0.5, random_state = 50)

# initialize a 3x1 matrix with random values as Weight Matrix
W = np.random.randn(X.shape[1],1)

# initialize a list for storing loss values during epochs
losses_value = []

# define sigmoid activation function
def sigmoid_activation_function(x):
    # calculate the sigmoid activation value
    return 1.0/(1+np.exp(-x))

# define predict function
def predict_function(X, W):
    prediction = sigmoid_activation_function(X.dot(W))
    # use step function to convert the prediction to class labels 0 or 1
    prediction[prediction <= 0.5] = 0
    prediction[prediction > 0.5] = 1
    return prediction

# define a method to split trainig data into batches
def get_batches(X, y, batch_size):
    # loop through the training data and yields the tuples which contain the batches of data
    for i in np.arange(0, X.shape[0], batch_size):
        yield((X[i:i+batch_size], y[i:i+batch_size]))

# starting the training epochs
print("Starting training epochs")
# Looping the number of epochs
for epoch in np.arange(0, no_of_epochs):
    # declare loss variable for every epoch (avg during every batch)
    loss_for_epoch = []
    
    # loop for every batch of training data
    for (batchX, batchY) in get_batches(X, y, batch_size):
        
        predictions = sigmoid_activation_function(batchX.dot(W))
        # to find the error, subtract the true output from the predicted output
        error = predictions - batchY
        # find the loss values and append it to the loss_for_epoch list
        loss_value = np.sum(error**2)
        loss_for_epoch.append(loss_value)
        # find the slope (gradient), dot product of the training input (transpose) and the error
        gradient = batchX.T.dot(error)
        # add to the existing value of Weight W, the new variation
        W += -(learning_rate) * gradient
    
    # find the average loss and add it to the global loss lost
    losses_value_average = np.average(loss_for_epoch)
    losses_value.append(losses_value_average)
    print("Epoch Number: {}, loss: {:.7f}".format(int(epoch+1),loss_value))
    
# starting the testing/evaluation
print("Staring testing/evaluation")
# obtain the predictions by using testing input data and the already computed and updated weight value W
predictions = predict_function(testX, W)
# give report by comparing the predictions and the truth value testY
print(classification_report(testY, predictions))
# print(W)

# plotting the dataset as scatter plot
plt.style.use("ggplot")
plt.figure()
plt.title("Scatter Plot of the data set")
plt.scatter(testX[:,0], testX[:,1])

# plotting error vs epoch graph
plt.style.use("ggplot")
plt.figure()
plt.title("Error vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(np.arange(0, no_of_epochs), losses_value)
