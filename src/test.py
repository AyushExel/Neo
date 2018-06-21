'''
This file contains the tests for evaluating the functions in nn.py
'''

import sklearn.datasets as datasets
import nn
import numpy as np
import pandas as pd
import optimizer

def test_run():
    """
    Sample test run.

    :return: None
    """
    # test run for binary classification problem:
    np.random.seed(3)
    print('Running a binary classification test')

    #Generate sample binary classification data
    data = datasets.make_classification(n_samples=200,n_features=10,n_classes=2)
    X= data[0].T
    Y = (data[1].reshape(200,1)).T
    net = nn.nn([10,20,1],['relu','sigmoid'])
    net.cost_function = 'CrossEntropyLoss'
    print('net architecture :')
    print(net)
    #Optimize using standard gradient descenet
    optim = optimizer.gradientDescentOptimizer
    optim(X,Y,net,alpha=0.07,epoch=200,lamb=0.05,print_at=100)
    output = net.forward(X)
    #Convert the probabilities to output values
    output = 1*(output>=0.5)
    accuracy = np.sum(output==Y)/200
    print('for gradient descenet \n accuracy = ' ,accuracy*100)

    #Optimize using standard gradient descenet
    net = nn.nn([10,20,1],['relu','sigmoid'])
    net.cost_function = 'CrossEntropyLoss'
    optim = optimizer.SGDOptimizer
    optim(X,Y,net,16,alpha=0.07,epoch=5,lamb=0.05,print_at=1)
    output = net.forward(X)
    output = 1*(output>=0.5)
    accuracy = np.sum(output==Y)/200
    print('for stochaistic gradient descenet \n accuracy = ' ,accuracy*100)


    
    print('Running a regression test')
    #test run for regresssion problem:
    #Generate sample regression data
         
    X = np.random.randn(1,200)
    Y = X**2 
    net = nn.nn([1,10,1],['relu'])
    net.cost_function = 'MSELoss'
    print('net architecture :')
    print(net)

    #Optimize using standard gradient descenet
    print('for gradient descenet ')
    optim(X,Y,net,alpha=0.01,epoch=300,lamb=0.05,print_at=100)

    net = nn.nn([1,10,1],['relu'])
    net.cost_function = 'MSELoss'
    #Optimize using stochaistic gradient descenet
    print('for stochaistic gradient descenet ')
    optim(X,Y,net,alpha=0.3,epoch=20,lamb=0.05,print_at=1)
    




   

if __name__ == "__main__":
    test_run()

