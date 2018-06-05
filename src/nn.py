'''
Class containing functionality to build and train neural networks
Contains all activation and loss functions some other utility function.
'''

import numpy as np 

class nn:
    def __init__(self,layer_dimensions,activations):
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

        self.initialize_parameters(layer_dimensions)
        self.initialize_activations(activations)
    
    def initialize_parameters(self,layer_dimensions):
        '''
        Task -Random intialization of weights of a network described by given layer dimensions
        inputs- layer_dimensions => dimensions to layers of the network
        returns- none.
        '''
        for i in range(1,len(layer_dimensions)):
            self.parameters['W'+str(i)] = np.random.randn(layer_dimensions[i],layer_dimensions[i-1])*0.01
            self.parameters['b'+str(i)] = np.zeros((layer_dimensions[i],1))
    
    def initialize_activations(self,activations):
        '''
        Taks- Intialize the activation list 
        Inputs- A list dscribing various activation units
        Returns-none
        '''
        self.activations = activations
    
    def __linear_forward(self,A_prev,W,b):
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
        inputs- Z=>layer to activate, type=>type of activation function to be used
        returns- Activated layer and activation cache
        Note: This function treats 1 as starting index! first layer's index is 1.
        '''
        act_cache = [Z]
        act = None
        if((self.activations[n_layer-1]).lower()=='relu'):
            act = Z*(Z>0)
        if((self.activations[n_layer-1]).lower()=='tanh'):
            act = np.tanh(Z)
        if((self.activations[n_layer-1]).lower()=='sigmoid'):
            act = 1/(1+np.exp(-Z))
        
        #assert(act!=None)

        return act,act_cache
    
    def forward(self,input):
        '''
        Taks-To forward propagate the entire layer
        inputs- input=> Contains the input to the Network
        returns- Output of the network
        '''
        A = input

        for i in range(1,int(len(self.parameters)/2)):
            W = self.parameters['W'+str(i)]
            b = self.parameters['b'+str(i)]
            Z,linear_cache = self.__linear_forward(A,W,b)

            A,act_cache = self.__activate(Z,i)
            self.cache.append([linear_cache,act_cache])
        
        #For Last Layer
        W = self.parameters['W'+str(int(len(self.parameters)/2))]
        b = self.parameters['b'+str(int(len(self.parameters)/2))]
        Z,linear_cache = self.__linear_forward(A,W,b)
        if len(self.activations)==len(self.parameters)/2:
            A,act_cache = self.__activate(Z,len(self.activations))
            self.cache.append([linear_cache,act_cache])
        else:
            A = Z
            self.cache.append([linear_cache,[None]])

        return A

    def forward_upto(self,input,layer_num):
        '''
        Task- Calculates forward prop upto layer_num
        Inputs- input=> Contains the input to the Network
                layer_num=>Layer upto which forward prop is to be calculated
        Returns- Activations of layer layer_num
        '''
        if layer_num == int(len(self.parameters)/2):
            return self.forward(input)
        else:
            A = input
            for i in range(1,layer_num):
                W = self.parameters['W'+str(i)]
                b = self.parameters['b'+str(i)]
                Z,linear_cache = self.__linear_forward(A,W,b)

                A,act_cache = self.__activate(Z,i)
                self.cache.append([linear_cache,act_cache])
            return A
                

#test run:
data = np.random.randn(2,100)
net = nn([2,15,2],['tanh','relu'])
A = net.forward(data)
print(A.shape)

    

