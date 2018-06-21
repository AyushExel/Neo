'''
This class contains various optimizer functions in one place for better and modular understanding of overall library
'''
import nn
import numpy as np

def gradientDescentOptimizer(input,mappings,net:nn.nn,alpha=0.001,lamb=None, epoch=100,print_at=5,prnt=True):
    '''
    Performs gradient descent on the given network setting the default value of epoch and alpha if not provided otherwise

    :param input  : input for neural net
    :param mapping: Correct output of the function
    :param net    : nn.nn object which provides the network architecture
    :param alpha  : Learning rate
    :param lamb   : Regularization parameter
    :param epoch  : Number of iterations
    :param print_at: Print at multiples of 'print_at'
    :param prnt   : Print if prnt=true
    '''

    for i in range(epoch):
        net.cache = []
        prediction = net.forward(input)
        loss_function = (net.cost_function).lower()
        loss,regularization_cost = None,0
        if loss_function == 'mseloss':
            loss = net.MSELoss(prediction,mappings)
        if loss_function == 'crossentropyloss':
            loss = net.CrossEntropyLoss(prediction,mappings)

        if lamb != None:
            for params in range(len(net.cache)):
                regularization_cost = regularization_cost + np.sum(np.square(net.parameters['W'+str(params+1)]))
            regularization_cost = (lamb/(2*input.shape[1]))*regularization_cost
            
        loss = loss + regularization_cost
        net.lamb = lamb
        if prnt and i%print_at==0 :
            print('Loss at ',i, ' ' ,loss)
        net.backward(prediction,mappings)
        
        for l in range(int(len(net.parameters)/2)):
            net.parameters['W'+str(l+1)] = net.parameters['W'+str(l+1)] -alpha*net.grads['dW'+str(l+1)]
            net.parameters['b'+str(l+1)] = net.parameters['b'+str(l+1)] -alpha*net.grads['db'+str(l+1)]




    
