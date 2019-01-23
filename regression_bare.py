# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 11:27:07 2019

Linear regression example by simply using least squares error..
@author: NeilDG
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import dataset_generator as dg
from random import seed
from random import randrange

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

def visualize(X,Y, b0, b1):
    x_max = np.max(X) + 100
    x_min = np.min(X) - 100
    
    x = np.linspace(x_min, x_max, 1000)
    y = b0 + b1 * x
    
    plt.plot(x, y, color = "#00FF00", label = "Line")
    plt.scatter(X, Y, color = "#FF0000", label = "Data Point")

    plt.xlabel("X")
    plt.ylabel("Y")
    
    plt.legend()
    plt.show()

def regression(X,Y): 
    X = X.reshape(len(X),1)
    b = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    
    yhat = X.dot(b)
    
    return yhat,b

def computeRMSE(X,Y, b0, b1):
    rmse = 0;
    for i in range(len(X)):
        y_pred = b0 + b1 * X[i]
        rmse += (Y[i] - y_pred) ** 2
    
    rmse = np.sqrt(rmse/(len(X)))
    print("RMSE: ",rmse)

def column(matrix, i):
    return [row[i] for row in matrix]

def homework():
    size = 500
    data = dg.generateLinear(100, 1, 5, size)
    
    X = data[0]; Y = data[1]
    R = regression(X,Y)
    yhat = R[0]; b = R[1]
    
    plt.scatter(X,Y)
    plt.plot(X, yhat, color = "red")
    plt.show()
    
    
    b0 = np.mean(Y) - (b * np.mean(X))
    computeRMSE(X,Y, b0, b)  
    
    #cross-validation
    index = np.arange(size)
    folds = cross_validation_split(index)
    
    for i in range(len(folds)):
        data_fold = np.empty([2,len(folds[i])])
        
        #assemble data test
        for j in range(len(folds[i])):
            data_fold[0][j] = X[folds[i][j]]
            data_fold[1][j] = Y[folds[i][j]]
            
        #train dataset
        train_fold = []
        for k in range(len(folds)):
            if(k != i): 
                #assemble data subset
                for j in range(len(folds[k])):
                    value = []
                    value.append(X[folds[k][j]]); value.append(Y[folds[k][j]])
                    train_fold.append([X[folds[k][j]], Y[folds[k][j]]])
        
        M = np.array([column(train_fold,0), column(train_fold,1)])
        R = regression(M[0], M[1])
        yhat = R[0]; b = R[1]
        plt.scatter(column(train_fold, 0), column(train_fold, 1))
        plt.plot(column(train_fold, 0), yhat, color = "green",linewidth = 6)
        
        #test out the remaining
        plt.scatter(data_fold[0], data_fold[1])
        b0 = np.mean(data_fold[1]) - (b * np.mean(data_fold[0]))
        computeRMSE(X,Y, b0, b)  
        
        plt.savefig("figures/plt_with_fold_{}.png".format(i), bbox_inches='tight', dpi = 400)
        plt.show()

homework()