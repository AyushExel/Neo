'''
Class containing functionality to build and train neural networks
Contains all activation and loss functions some other utility function.
'''

import numpy as np 

class nn:
    def __init__(layer_dimensions,activations):
        '''
        Initializes networks's weights and other useful variables
        Variables-parameters=>contains weights of the layer in form {'Wi':[],'bi':[]}
                  cache=>contains intermediate results as [[A[i-1],Wi,bi],[Zi]],where i
                  is layer number.
                  activations=> TO store the activation for each layer
        '''
        self.parameters = {} 
        self.cache = []      
        self.activations= []

        initialize_parameters(layer_dimensions)
        initialize_activations(activations)
    
    def initialize_parameters(self,layer_dimensions):
        '''
        Task -Random intialization of weights of a network described by given layer dimensions
        inputs- layer_dimensions => dimensions to layers of the network
        returns- none.
        '''
        for i in range(1,len(layer_dimensions)):
            self.parameters['W'+str(i)] = np.random.randn(layer_dimensions[i],layer_dimensions[i-1])*0.01
            self.parameters['b'+str(i)] = np.zeros(layer_dimensions[i])
    
    def initialize_activations(self,activations):
        '''
        Taks- Intialize the activation list 
        Inputs- A list dscribing various activation units
        Returns-none
        '''
        self.activations = activations
    
    def __linear_forward(A_prev,W,b):
        '''
        Task- Linear forward to the current layer using previous activations
        inputs- A_prev=>Previous Layer's activation,
                W,b=> Weights and biases for current layer
        returns- linear cache and current calculated layer
        '''
        Z = W.dot(A_prev) + b
        linear_cache= [A_prev,W,b]
        return Z,linear_cache

    def __activate(self,Z,n_layer):
        '''
        Task-Activate the given layer(Z) using the activation function specified by 'type'
        inputs- Z=>layer to activate, type=>type of activation function to be used.
        returns- Activated layer and activation cache
        '''
        act_cache = [Z]
        act = None
        if(lower(self.activations[n_layer-1])=='relu'):
            act = 1*np.max(0,Z)
        if(lower(self.activations[n_layer-1])=='tanh'):
            act = np.tanh(Z)
        if(lower(self.activations[n_layer-1])=='sigmoid'):
            act = 1/(1+np.exp(-Z))
        
        assert(act!=None)

        return act,act_cache
    
    #def forward():

    

