'''
Class containing functionality to build and train neural networks
Contains all activation and loss functions some other utility function.
'''

import numpy as np 

class nn:
    def __init__(layer_dimensions):
        '''
        Initializes networks's weights and other useful variables
        Variables-parameters=>contains weights of the layer in form {'Wi':[],'bi':[]}
                  cache=>contains intermediate results as ['activation',A[i-1],Wi,bi,Zi],where i
                  is layer number.
        '''
        self.parameters = {} 
        self.cache = []      
        initialize_parameters(layer_dimensions)
    
    def initialize_parameters(self,layer_dimensions):
        '''
        Task -Random intialization of weights of a network described by given layer dimensions
        inputs- layer_dimensions => dimensions to layers of the network
        returns- none.
        '''
        for i in range(1,len(layer_dimensions)):
            self.parameters['W'+str(i)] = np.random.randn(layer_dimensions[i],layer_dimensions[i-1])*0.01
            self.parameters['b'+str(i)] = np.zeros(layer_dimensions[i])
    def activate(self,Z,type):
        '''
        Task-Activate the given layer(Z) using the activation function specified by 'type'
        inputs- Z=>layer to activate, type=>type of activation function to be used.
        '''
    

