# -*- coding: utf-8 -*-
"""
Utility functions
Created on Fri Feb  1 21:01:23 2019

@author: NeilDG
"""

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

def column(matrix, i):
    return [row[i] for row in matrix]