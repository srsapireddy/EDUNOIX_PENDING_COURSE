# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:37:39 2020

@author: Rahul Sapireddy
"""

# importing the basic packages
import numpy as np

# declaring the class and constructor
class BackPropagation:
    def __init__(self, layers_list, learning_rate=0.1):
        # have an empty list for weight matrix
        self.W = []
        # set the learning rate
        self.learning_rate = learning_rate
        # set the layers for network
        self.layers_list = layers_list
        
        # for loop for generating the weight matrix for inner layer except the last two
        for i in np.arange(0, len(layers_list) - 2):
            # generate random value for weight matrix
            w = np.random.randn(layers_list[i]+1, layers_list[i+1]+1)
            self.W.append(w / np.sqrt(layers_list[i]))

        # generate the weight matrix for the last two layers
        w = np.random.randn(layers_list[-2] + 1, layers_list[-1]) 
        self.W.append(w / np.sqrt(layers_list[-2]))
        
        
    # define sigmoid activation function
    def sigmoid_activation_function(self, x):
        # calculate the sigmoid activation value
        return 1.0/(1+np.exp(-x))
    
    # define derivative of sigmoid activation function
    def deriv_sigmoid_activation_function(self, x):
        # calculate the sigmoid activation value
        return x*(1-x)
    
    # define the train_fit method -> for the weight update
    def train_fit(self, X, y, no_of_epochs=1000):
        # add a one column to the input feature matrix
        X = np.c_[X, np.ones((X.shape[0]))]

        # loop through every epoch
        for single_epoch in np.arange(0, no_of_epochs):
            # loop through every data point
            for (training_input, expected_output) in zip(X, y):
                # PENDING
                # DEFINE SEPERATE METHOD FOR PARTIAL FIT
                self.fit_fwd_bwd_partial(training_input, expected_output)
                # PENDING OPTION TO DISPLAY THE PROGRESS OF TRAINING
                # DISPLAY CURRENT ERRROR STATUS
                loss = self.find_loss(training_input, expected_output)
                if(single_epoch == 0 or (single_epoch+1) % 100): 
                    print("epoch no = {}, loss = {:.7f}".format(single_epoch+1, loss))
                
    # defining the partial fit method for fwd pass and bwd propagation
    def fit_fwd_bwd_partial(self, x, y):
        # for the first layer the activations will be the data item itself (Eg: 0, 0, 1)
        Layer_Activations = [np.atleast_2d(x)]
        
        # Feed forward Mechanism
        # Loop through the rest of the layers and find the activation for these layers
        # Find the dot product
        for individual_layer in np.arange(0, len(self.W)):
            # The length of weight matrix will give the number of nodes in the particular layer
            # step 1: find the dot product of the input feature and the weight
            dot_product = Layer_Activations[individual_layer].dot(self.W[individual_layer])
            # step 2: pass the dot product to the sigmoid function
            sigmoid_out = self.sigmoid_activation_function(dot_product)
            # step 3: append obtained value to the list of activations
            Layer_Activations.append(sigmoid_out)
            
        # backprapogation mechanism
        output_error = Layer_Activations[-1] - y
        # find the delta of the last layer and add it to the list of delta
        Delta_List = [output_error * self.deriv_sigmoid_activation_function(Layer_Activations[-1])] 
        # Loop through the rest of the layers in reverse and find the deltas for these layers
        for individual_layer in np.arange(len(Layer_Activations)-2,0,-1):
            # delta for current layer - delta of previous layer 
            # (Transpose) Weight matrix of current layer *
            # [sigmoid_derivative(Current Layer activation)]
            delta_value = Delta_List[-1].dot(self.W[individual_layer].T)
            delta_value = delta_value * self.deriv_sigmoid_activation_function(Layer_Activations[-1])
            Delta_List.append(delta_value)
            
        # reverse the list of deltas
        Delta_List = Delta_List[::-1]
        # Loop through every layer and update the weight
        for individual_layer in np.arange(0, len(self.W)):
            # update the weights for current layer
            self.W[individual_layer] += -self.learning_rate * Layer_Activations[individual_layer].T.dot(Delta_List[individual_layer])
            
    # define the predict_evaluation method
    def predict_eval(self, X):
        # convert to 2d matrix if its not 2d
        X = np.atleast_2d(X)
        # add a one column to the input feature matrix
        prediction = np.c_[X, np.ones((X.shape[0]))]
        # find the prediction and return it
        # Feed forward Mechanism
        # Loop through the every layer and find the predictions of those layers
        # Find the dot product
        for individual_layer in np.arange(0, len(self.W)):
            # The length of weight matrix will give the number of nodes in the particular layer
            # step 1: find the dot product of the input feature and the weight
            prediction = prediction.dot(self.W[individual_layer])
            # step 2: pass the dot product to the sigmoid function
            prediction = self.sigmoid_activation_function(prediction)
            
        # return the prediction
        return prediction
    
    # define the find_loss method
    def find_loss(self, X, Y):
        # convert to 2d matrix if its not 2d
        actual_output = np.atleast_2d(X)
        # find the prediction
        predicted_output = self.predict_eval(X)
        # find the error
        calculated_error = predicted_output - actual_output
        loss_value = 0.5 * np.sum(calculated_error ** 2)
        return loss_value
            
        

















