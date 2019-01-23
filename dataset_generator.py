# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 14:39:15 2019

@author: NeilDG
"""
import numpy as np
import matplotlib.pyplot as plt

def generateLinear(base = 1, a = 1, b = 1, samples = 100):
    mu, sigma = 0, 0.5
    S = np.random.normal(mu,sigma, samples) * base;
    
    X = np.arange(samples)
    Y = (a * X) + (b*S)
    
    return X,Y