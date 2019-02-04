# -*- coding: utf-8 -*-
"""
Script for linear perceptron and MLP
Created on Fri Feb  1 20:56:00 2019

@author: NeilDG
"""
from random import seed
from random import randrange
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import utility as util
import linear_perceptron as lp
import multilayer_perceptron as mlp

def cross_validation_split(dataset, folds=10):
    seed(1)
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
            
        dataset_split.append(fold)
    return dataset_split

def lp_test(dataset):
    size = np.arange(np.size(dataset, axis = 0))
    folds = cross_validation_split(size)
    
    sum_accuracy = 0.0

    for i in range(np.size(folds, axis = 0)):
        train = np.empty([0, np.size(dataset, axis = 1)], dtype = np.object)
        
        #assemble training data
        for j in range(np.size(folds,axis = 0)):
            if i != j:
                for k in range(np.size(folds[j], axis = 0)):
                    ind = folds[j][k]
                    train = np.insert(train, 0, values = dataset[ind], axis = 0)
    
        linearPerceptron = lp.LinearPerceptron(train, learning_rate = 0.0001, epochs = 500)
        linearPerceptron.learn()
        
        #assemble test
        test = np.empty([0, np.size(dataset, axis = 1)], dtype = np.object)
        for k in range(np.size(folds[i], axis = 0)):
            ind = folds[i][k]
            test = np.insert(test, 0, values = dataset[ind], axis = 0)
        
        
        sum_accuracy += linearPerceptron.evaluate(test)
        
    total_accuracy = sum_accuracy / np.size(folds, axis = 0) * 1.0
    print("Total accuracy using 10-fold cross validation: ", total_accuracy)

def mlp_test(dataset):
    lr = 1.0; epochs = 500
    print("[MLP] Training data with learning rate: " ,lr, " Epochs: " ,epochs)
    
    size = np.arange(np.size(dataset, axis = 0))
    folds = cross_validation_split(size)
    
    sum_accuracy = 0.0

    for i in range(np.size(folds, axis = 0)):
        train = np.empty([0, np.size(dataset, axis = 1)], dtype = np.object)
        
        #assemble training data
        for j in range(np.size(folds,axis = 0)):
            if i != j:
                for k in range(np.size(folds[j], axis = 0)):
                    ind = folds[j][k]
                    train = np.insert(train, 0, values = dataset[ind], axis = 0)
    
        multilayerPerceptron = mlp.MultilayerPerceptron(train, learning_rate = lr, epochs = epochs)
        multilayerPerceptron.learn()
        
        #assemble test
        test = np.empty([0, np.size(dataset, axis = 1)], dtype = np.object)
        for k in range(np.size(folds[i], axis = 0)):
            ind = folds[i][k]
            test = np.insert(test, 0, values = dataset[ind], axis = 0)
        
        
        sum_accuracy += multilayerPerceptron.evaluate(test)
        
    total_accuracy = sum_accuracy / np.size(folds, axis = 0) * 1.0
    print("Total accuracy using 10-fold cross validation: ", total_accuracy)
    
def main(): 
    #load iris dataset
    raw = pd.read_csv("dataset/iris.csv")
    dataset = raw.values
    
    #lp_test(dataset)
    mlp_test(dataset)
    
    

main()

