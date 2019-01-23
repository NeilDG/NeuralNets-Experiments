# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 18:01:10 2019

Sample and experiments from net
@author: NeilDG
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def regression_1():
    dataset = pd.read_csv("dataset/lr_sample.csv")
    print(dataset.shape)
    dataset.head()
    
    #initialize inputs and ouputs
    X = dataset['Head Size(cm^3)'].values
    Y = dataset['Brain Weight(grams)'].values
    
    #get mean
    x_mean = np.mean(X)
    y_mean = np.mean(Y)
    
    n = len(X);
    
    #calculate b1 and b0
    num = 0; den = 0;
    for i in range(n):
        num += (X[i] - x_mean) * (Y[i] - y_mean)
        den += (X[i] - x_mean) ** 2
    
    
    b1 = num / den
    b0 = y_mean - (b1 * x_mean)
    
    print("Output is: ", b1, b0)


def regression_2():
    dataset = pd.read_csv("dataset/lr_sample.csv")
    print(dataset.shape)
    dataset.head()
    
    #initialize inputs and ouputs
    X = dataset['Head Size(cm^3)'].values
    Y = dataset['Brain Weight(grams)'].values
    
    X = X.reshape(len(X),1)
    #linear least squares
    b = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    print(b)
    
    yhat = X.dot(b)
    
    plt.scatter(X,Y)
    plt.plot(X, yhat, color = "red")
    plt.show()
    
    b0 = np.mean(Y) - (b * np.mean(X))

    
def regression_3():
    dataset = pd.read_csv("dataset/lr_sample.csv")
    print(dataset.shape)
    dataset.head()
    
    #initialize inputs and ouputs
    X = dataset['Head Size(cm^3)'].values
    Y = dataset['Brain Weight(grams)'].values
    
    a_0 = len(X); a_1 = np.sum(X); a_2 = a_1; a_3 = a_1 ** 2
    A = np.matrix([[a_0, a_1], [a_2, a_3]])
    y = np.matrix([[np.sum(Y)],[np.sum(Y * X)]])
    print("A: ", A) 
    print("Y: " ,y)
    
    
    w = np.linalg.inv(A) * y
    print("A inverse: ", np.linalg.inv(A))
    print("W:" ,w.item(0), w.item(1))
    
    yhat = (w.item(1) * X) + w.item(0);
    
    plt.scatter(X,Y)
    plt.plot(X, yhat, color = "red")
    plt.show()
    
    b0 = np.mean(Y) - (w.item(0) * np.mean(X))

