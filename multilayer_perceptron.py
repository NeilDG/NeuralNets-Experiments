# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 20:44:40 2019

Class representing a multilayer perceptron
@author: NeilDG
"""

import numpy as np

class MultilayerPerceptron(object):
    
    def __init__(self, training_data, learning_rate = 0.001, epochs = 10, class_index = -1, class_size = 3):
        self.training_data = training_data
        self.class_index = class_index
        self.epochs = epochs
        self.lr = learning_rate
        
        #remove class for training
        self.X = np.delete(self.training_data, class_index, axis = 1)
        self.X = np.insert(self.X, 0, values = 1, axis = 1) #insert bias
        
        self.Z = np.empty(np.size(self.X, axis = 1)) #create Z (hidden) vector based from X TODO: make this a hyperparameter
        
        #weight vector to Z initialize
        train_col = np.size(self.X, axis = 1)
        self.W = np.array([np.random.random(train_col)])
        for i in range(np.size(self.Z, axis = 0)):
            self.W = np.insert(self.W, 0, values = np.random.random(train_col), axis = 0)
            
        
        #Y output initialize
        self.Y = np.zeros(class_size)
        
        #weight vector to Y initialize
        self.V = np.array([np.random.random(train_col)])
        for i in range(np.size(self.Y, axis = 0)):
            self.V = np.insert(self.V, 0, values = np.zeros(train_col), axis = 0)
        
        #print("Z: ", np.size(self.Z), "Y: ", np.shape(self.Y), "V: ", np.shape(self.V), "W: ", np.shape(self.W))
 
    #hardcoded way of identifying class by number
    def identifyClass(self, class_string):
        if class_string == "setosa":
            return 0
        elif class_string == "versicolor":
            return 1
        elif class_string == "virginica":
            return 2
        
    def output(self, Y): 
        index = np.argmax(Y);
        return index
    
    def learn(self):
        d = np.size(self.X, axis = 1)
        k = np.size(self.Y)
        
        
        for epoch in range(self.epochs):
            sum_error = 0.0
            
            for t in range(np.size(self.X, axis = 0)):
                #compute sigmoid for hidden layer
                A = np.zeros(np.size(self.Z, axis = 0))
                for i in range(k):
                    A[i] += self.W[i,0]
                    for j in range(d - 1):
                        A[i] += (self.W[i,j] * self.X[t,j])
                        
                for i in range(k):
                    self.Z[i] = 1 / (1 + np.exp(-A[i]))
                
                
                #compute softmax for Y
                for i in range(k):
                    O = np.zeros(np.size(self.Y, axis = 0))
                    self.Y[i] += self.V[i, 0]
                    for h in range(d - 1):
                        O[i] += (self.V[i,h + 1] * self.Z[i])
                
                for i in range(k):
                     self.Y[i] = np.exp(O[i]) / np.sum(np.exp(O))
                
                #print("Y: ", self.Y, "Sum of Y: " ,np.sum(self.Y))
                 
                
                delta_v = np.empty(np.shape(self.V))
                w_error = 0.0
                for i in range(k):
                    for j in range(d):
                        pred = self.output(self.Y)
                        label = self.training_data[t,self.class_index]
                        actual = self.identifyClass(label)
                        r = 1.0 if pred == actual else 0.0
                        error = (r - self.Y[i])
                        sum_error += error
                        w_error += error
                        delta_v[i,j] =  self.lr * error * self.Z[i]
                
                #print("Delta V: " ,np.size(delta_v))
                
                delta_w = np.empty(np.shape(self.W))
                for i in range(k):
                    for j in range(d):
                        delta_w[i,j] = (self.lr * w_error * delta_v[i,j]) * (self.Z[i] * (1 - self.Z[i]) * self.X[t,j])
                
                #print("Delta W: ", delta_w)
                
                for i in range(k):
                    for j in range(d):
                        self.V[i,j] = self.V[i,j] + delta_v[i,j]
                
                for i in range(k):
                    for j in range(d):
                        self.W[i,j] = self.W[i,j] + delta_w[i,j]
                
                
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
                self.Z[i] = np.exp(A[i]) / np.sum(np.exp(A))
            
            for i in range(k):
                total = self.V[i, 0]
                for h in range(d - 1):
                    total += (self.V[i,h + 1] * self.Z[i])
                
                self.Y[i] = total
                    

            pred = self.output(self.Y)
            label = T[t,class_index]
            actual = self.identifyClass(label)
            if  pred != actual:
                sum_error+=1
            
        
        accuracy = ((sum_error/t * 1.0))*100.0
        print("[MLP] Total errors: " ,sum_error, " Test size: " ,t, " Accuracy: " ,accuracy)
        return accuracy
