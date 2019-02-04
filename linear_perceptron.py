# -*- coding: utf-8 -*-
"""
Class representing a linear perceptron. Also trying out how OOP works for Python.
Created on Fri Feb  1 21:10:21 2019

@author: NeilDG
"""

import numpy as np

class LinearPerceptron(object):
    #A linear perceptron
    
    def __init__(self, training_data, learning_rate = 0.001, epochs = 10, class_index = -1, class_size = 3):
        self.training_data = training_data
        self.class_index = class_index
        self.epochs = epochs
        self.lr = learning_rate
        
        #remove class for training
        self.X = np.delete(self.training_data, class_index, axis = 1)

        #weight vector with bias
        train_col = np.size(self.X, axis = 1) + 1
        self.W = np.array([np.random.random(train_col)])
        for i in range(class_size):
            self.W = np.insert(self.W, 0, values = np.random.random(train_col), axis = 0)
        
        self.X = np.insert(self.X, 0, values = 1, axis = 1) #insert bias
        self.Y = np.zeros(class_size)
      
        #print("LP parameters: ",self.W, " Epochs: " ,self.epochs, " Learning rate: " ,self.lr)

    #hardcoded way of identifying class by number
    def identifyClass(self, class_string):
        if class_string == "setosa":
            return 0
        elif class_string == "versicolor":
            return 1
        elif class_string == "virginica":
            return 2
    
    #activation function
    def output(self, Y): 
        index = np.argmax(Y);
        return index
                
    def learn(self):
        d = np.size(self.X, axis = 1)
        k = np.size(self.Y)
        
        
        for epoch in range(self.epochs):
            sum_error = 0.0
            
            for t in range(np.size(self.X, axis = 0)):
                A = np.zeros(np.size(self.Y, axis = 0))
                for i in range(k):
                    for j in range(d):
                        A[i] += self.W[i,j] * self.X[t,j]
                        
                for i in range(k):
                    self.Y[i] = np.exp(A[i]) / np.sum(np.exp(A))
                #print("Y: ", self.Y, "Sum of Y: " ,np.sum(self.Y))
                    
                for i in range(k):
                    for j in range(d):
                        pred = self.output(self.Y)
                        label = self.training_data[t,self.class_index]
                        actual = self.identifyClass(label)
                        r = 1.0 if pred == actual else 0.0
                        error = (r - self.Y[i])
                        sum_error += error
                        self.W[i,j] = self.W[i,j] + self.lr * error * self.X[t,j]
            
            #print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, self.lr, sum_error))
                     
   
    def evaluate(self, T, class_index = -1):
        #remove class for testing
        T_hat = np.delete(T, class_index, axis = 1)
        d = np.size(T, axis = 1)
        k = np.size(self.Y)
        
        sum_error = 0.0;
        
        for t in range (np.size(T_hat, axis = 0)):
            A = np.zeros(np.size(self.Y, axis = 0))
            for i in range(k):
                for j in range(d):
                    A[i] += self.W[i,j] * self.X[t,j]
                        
            for i in range(k):
                self.Y[i] = np.exp(A[i]) / np.sum(np.exp(A))
                    

            pred = self.output(self.Y)
            label = T[t,class_index]
            actual = self.identifyClass(label)
            if  pred != actual:
                sum_error+=1
            
        
        accuracy = ((sum_error/t * 1.0))*100.0
        print("[LP] Total errors: " ,sum_error, " Test size: " ,t, " Accuracy: " ,accuracy)
        return accuracy
                
            
    # Make a prediction with weights
    def output_old(self, row):
    	activation = self.W[0]
    	for i in range(len(row)-1):
    		activation += self.W[i + 1] * row[i]
    	return 1.0 if activation >= 0.0 else 0.0

    # Estimate Perceptron weights using stochastic gradient descent
    def learn_old(self, T, Y):
        print("Y:", Y)
        for epoch in range(self.epochs):
            sum_error = 0.0
            for row in T:
                prediction = self.output_old(row)
                error = Y - prediction
                sum_error += error**2
                self.W[0] = self.W[0] + self.lr * error
                for i in range(len(row)-1):
                    self.W[i + 1] = self.W[i + 1] + self.lr * error * row[i]
                
                print('>epoch=%d, lrate=%.3f, error=%.3f' % (self.epochs, self.lr, sum_error))
        
        return self.W
