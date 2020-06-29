# -*- coding: utf-8 -*-
"""
Created on Wed May 20 01:21:52 2020

@author: Rahul Sapireddy
"""

# import the required libraries and classes
import numpy as np
from custom_perceptron import SimplePerceptron

# create the AND dataset
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0],[0],[0],[1]])

# print(X.shape)
# print(y.shape)

# initialize the perceptron
perceptron = SimplePerceptron(X.shape[1], learning_rate=0.1)

# train the perceptron using train_fit method
print("Training the perceptron for AND")
perceptron.train_fit(X, y, no_of_epochs=20)

# evaluate the prediction done by the perceptron
print("Starting the perceptron for AND")

# loop through the dataset
for(input_data, actual_output) in zip(X, y):
    # get the predicted output
    sys_prediction = perceptron.predict_eval(input_data)
    print("for inpput {}, the predicted is {}, sctual is {}".format(input_data, sys_prediction, int(actual_output)))
    
# create the OR dataset
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0],[1],[1],[1]])

# print(X.shape)
# print(y.shape)

# initialize the perceptron
perceptron = SimplePerceptron(X.shape[1], learning_rate=0.1)

# train the perceptron using train_fit method
print("Training the perceptron for OR")
perceptron.train_fit(X, y, no_of_epochs=20)

# evaluate the prediction done by the perceptron
print("Starting the perceptron for OR")

# loop through the dataset
for(input_data, actual_output) in zip(X, y):
    # get the predicted output
    sys_prediction = perceptron.predict_eval(input_data)
    print("for inpput {}, the predicted is {}, sctual is {}".format(input_data, sys_prediction, int(actual_output)))
    
# create the XOR dataset
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0],[1],[1],[0]])

# print(X.shape)
# print(y.shape)

# initialize the perceptron
perceptron = SimplePerceptron(X.shape[1], learning_rate=0.1)

# train the perceptron using train_fit method
print("Training the perceptron for XOR")
perceptron.train_fit(X, y, no_of_epochs=20)

# evaluate the prediction done by the perceptron
print("Starting the perceptron for XOR")

# loop through the dataset
for(input_data, actual_output) in zip(X, y):
    # get the predicted output
    sys_prediction = perceptron.predict_eval(input_data)
    print("for inpput {}, the predicted is {}, sctual is {}".format(input_data, sys_prediction, int(actual_output)))
