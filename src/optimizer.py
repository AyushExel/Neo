'''
This class contains various optimizer functions in one place for better and modular understanding of overall library
'''
import nn
import numpy as np

class optimizer:
    @staticmethod
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

    @staticmethod
    def SGDOptimizer(input,mappings,net:nn.nn,mini_batch_size=64,alpha=0.001,lamb=None, epoch=5,print_at=5,prnt=True):
        '''
        Performs Stochaitic gradient descent on the given network
        -Generates mini batches of given size using random permutation
        -Uses gradient descent on each mini batch separately

        :param input  : input for neural net
        :param mapping: Correct output of the function
        :param net    : nn.nn object which provides the network architecture
        :param batch_size: Batch size to be used witch SGD
        :param alpha  : Learning rate
        :param lamb   : Regularization parameter
        :param epoch  : Number of iterations
        :param print_at: Print at multiples of 'print_at'
        :param prnt   : Print if prnt=true
        '''
        batch_size = input.shape[1]
        mini_batches = []
        
        permutation = list(np.random.permutation(batch_size))
        shuffled_input = input[:,permutation]
        shuffled_mappings = (mappings[:,permutation])

        num_complete_batches = int(np.floor(batch_size/mini_batch_size))
        
        #Separate the complete mini_batches
        for i in range(0,num_complete_batches):
            mini_batch_input = shuffled_input[:,i*mini_batch_size:(i+1)*mini_batch_size]
            mini_batch_mappings = shuffled_mappings[:,i*mini_batch_size:(i+1)*mini_batch_size]
            mini_batch = (mini_batch_input,mini_batch_mappings)
            mini_batches.append(mini_batch)
        
        #Separate the incomplete mini batch if any
        if batch_size % mini_batch_size != 0:
            mini_batch_input = shuffled_input[:,batch_size - num_complete_batches*mini_batch_size : batch_size]
            mini_batch_mappings = shuffled_mappings[:,batch_size - num_complete_batches*mini_batch_size : batch_size]
            mini_batch = (mini_batch_input,mini_batch_mappings)
            mini_batches.append(mini_batch)
        
        for i in range(1,epoch+1):
            for batches in range(len(mini_batches)):
                optimizer.gradientDescentOptimizer(input,mappings,net,alpha,lamb,epoch=1,prnt=False)
            prediction = net.forward(input)
            loss = None 
            loss_function = net.cost_function.lower()
            if loss_function == 'mseloss':
                loss = net.MSELoss(prediction,mappings)
            if loss_function == 'crossentropyloss':
                loss = net.CrossEntropyLoss(prediction,mappings)
            
            if i%print_at == 0:
                print('Loss at ', i , ' ' , loss)
        
    

        


    


    
