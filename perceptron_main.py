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


def main(): 
    #load iris dataset
    raw = pd.read_csv("dataset/iris.csv")
    dataset = raw.values
    
    linearPerceptron = lp.LinearPerceptron(dataset, learning_rate = 0.001, epochs = 500)
    linearPerceptron.learn()
    

main()

