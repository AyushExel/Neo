"""
Class containing functionality to build and train neural networks
Contains all activation and loss functions some other utility function.
"""

import numpy as np


class nn:
    def __init__(self, layer_dimensions, activations):
        """
        Initializes networks's weights and other useful variables.

        :param layer_dimensions:
        :param activations: To store the activation for each layer
        -Parameters contains weights of the layer in form {'Wi':[],'bi':[]}
        -Cache contains intermediate results as [[A[i-1],Wi,bi],[Zi]], where i
         is layer number.
        -activations contains the names of activation function used for that layer
        -cost_function  contains the name of cost function to be used
        -lamb contains the regularization hyper-parameter
        -grads contains the gradients calculated during back-prop in form {'dA(i-1)':[],'dWi':[],'dbi':[]}
        -layer_type contains the info about the type of layer( fc, conv etc)
        """
        self.parameters = {}
        self.cache = []
        self.activations = activations
        self.cost_function = ''
        self.lamb = 0
        self.grads = {}
        self.layer_type = []
        
        self.initialize_parameters(layer_dimensions)
        self.check_activations()
 

    def initialize_parameters(self, layer_dimensions):
        """
        Xavier initialization of weights of a network described by given layer
        dimensions.

        :param layer_dimensions: Dimensions to layers of the network
        :return: None
        """
        for i in range(1, len(layer_dimensions)):
            self.parameters["W" + str(i)] = (
                np.sqrt(2/layer_dimensions[i - 1])*np.random.randn(layer_dimensions[i],
                                layer_dimensions[i - 1])
            )
            self.parameters["b" + str(i)] = np.zeros((layer_dimensions[i], 1))
            self.layer_type.append('fc')
    
    def check_activations(self):
        '''
        Checks if activations for all layers are present. Adds 'None' if no activations are provided for a particular layer.
        
        :returns: None
        '''
        num_layers = int(len(self.parameters)/2)
        while len(self.activations) < num_layers :
            self.activations.append(None)
        

    @staticmethod
    def __linear_forward(A_prev, W, b):
        """
        Linear forward to the current layer using previous activations.

        :param A_prev: Previous Layer's activation
        :param W: Weights for current layer
        :param b: Biases for current layer
        :return: Linear cache and current calculated layer
        """
        Z = W.dot(A_prev) + b
        linear_cache = [A_prev, W, b]
        return Z, linear_cache

    def __activate(self, Z, n_layer=1):
        """
        Activate the given layer(Z) using the activation function specified by
        'type'.

        Note: This function treats 1 as starting index!
              First layer's index is 1.

        :param Z: Layer to activate
        :param n_layer: Layer's index
        :return: Activated layer and activation cache
        """
        act_cache = [Z]
        act = None
        if (self.activations[n_layer - 1]) == None:
            act = Z
        elif (self.activations[n_layer - 1]).lower() == "relu":
            act = Z * (Z > 0)
        elif (self.activations[n_layer - 1]).lower() == "tanh":
            act = np.tanh(Z)
        elif (self.activations[n_layer - 1]).lower() == "sigmoid":
            act = 1 / (1 + np.exp(-Z))
        elif (self.activations[n_layer - 1]).lower() == "softmax":
            act = np.exp(Z-np.max(Z))
            act = act/(act.sum(axis=0)+1e-10)
        

        # assert(act!=None)

        return act, act_cache

    def forward(self, net_input):
        """
        To forward propagate the entire Network.

        :param net_input: Contains the input to the Network
        :return: Output of the network
        """
        self.cache = [] 
        A = net_input

        for i in range(1, int(len(self.parameters) / 2)):
            W = self.parameters["W" + str(i)]
            b = self.parameters["b" + str(i)]
            Z = linear_cache = None
            if self.layer_type[i] == 'fc':
                Z, linear_cache = self.__linear_forward(A, W, b)
            A, act_cache = self.__activate(Z, i)

            self.cache.append([linear_cache, act_cache])

        # For Last Layer
        W = self.parameters["W" + str(int(len(self.parameters) / 2))]
        b = self.parameters["b" + str(int(len(self.parameters) / 2))]
        Z, linear_cache = self.__linear_forward(A, W, b)
        if len(self.activations) == len(self.parameters) / 2:
            A, act_cache = self.__activate(Z, len(self.activations))
            self.cache.append([linear_cache, act_cache])
        else:
            A = Z
            self.cache.append([linear_cache, [None]])

        return A

    def forward_upto(self, net_input, layer_num):
        """
        Calculates forward prop upto layer_num.

        :param net_input: Contains the input to the Network
        :param layer_num: Layer up to which forward prop is to be calculated
        :return: Activations of layer layer_num
        """
        if layer_num == int(len(self.parameters) / 2):
            return self.forward(net_input)
        else:
            A = net_input
            for i in range(1, layer_num):
                W = self.parameters["W" + str(i)]
                b = self.parameters["b" + str(i)]
                Z, linear_cache = self.__linear_forward(A, W, b)

                A, act_cache = self.__activate(Z, i)
                self.cache.append([linear_cache, act_cache])
            return A
    
    def MSELoss(self,prediction,mappings):
        '''
        Calculates the Mean Squared error with regularization cost(if provided) between output of the network and the real
        mappings of a function.
        Changes cost_function to appropriate value

        :param prediction: Output of the neural net
        :param mappings: Real outputs of a function
        :return: Mean squared error b/w output and mappings
        '''

        self.cost_function = 'MSELoss'
        loss = np.square(prediction-mappings).mean()/2
        regularization_cost = 0
        if self.lamb != 0:
            for params in range(len(self.cache)):  
                regularization_cost = regularization_cost + np.sum(np.square(self.parameters['W'+str(params+1)]))
        regularization_cost = (self.lamb/(2*prediction.shape[1]))*regularization_cost
        
        return loss + regularization_cost

    def CrossEntropyLoss(self,prediction,mappings):
        '''
        Calculates the cross entropy loss between output of the network and the real mappings of a function
        Changes cost_function to appropriate value

        :param prediction: Output of the neural net
        :param mappings: Real outputs of a function
        :return: Mean squared error b/w output and mappings
        '''
        epsilon = 1e-8
        self.cost_function = 'CrossEntropyLoss'
        loss = -(1/prediction.shape[1])*np.sum( mappings*np.log(prediction+epsilon) + (1-mappings)*np.log(1-prediction+epsilon) )
        regularization_cost = 0
        if self.lamb != 0:
            for params in range(len(self.cache)):
                regularization_cost = regularization_cost + np.sum(np.square(self.parameters['W'+str(params+1)]))
        regularization_cost = (self.lamb/(2*prediction.shape[1]))*regularization_cost

        return loss + regularization_cost
    
    def output_backward(self,prediction,mapping):
        '''
        Calculates the derivative of the output layer(dA)

        :param prediction: Output of neural net
        :param mapping: Correct output of the function
        :param cost_type: Type of Cost function used
        :return: Derivative of output layer, dA  
        '''
        dA = None
        cost = self.cost_function
        if cost.lower() == 'crossentropyloss':
            dA =  -(np.divide(mapping, prediction+1e-10) - np.divide(1 - mapping, 1 - prediction+1e-10))
        
        elif cost.lower() == 'mseloss':   
            dA =  (prediction-mapping)
        
        return dA
    
    def deactivate(self,dA,n_layer):
        '''
        Calculates the derivate of dA by deactivating the layer

        :param dA: Activated derivative of the layer
        :n_layer: Layer number to be deactivated
        :return: deact=> derivative of activation 
        '''
        act_cache = self.cache[n_layer-1][1]
        dZ = act_cache[0]
        deact = None
        if self.activations[n_layer - 1] == None:
            deact = 1
        elif (self.activations[n_layer - 1]).lower() == "relu":
            deact = 1* (dZ>0)
        elif (self.activations[n_layer - 1]).lower() == "tanh":
            deact = 1- np.square(dA)
        elif (self.activations[n_layer - 1]).lower() == "sigmoid" or (self.activations[n_layer - 1]).lower()=='softmax':
            s = 1/(1+np.exp(-dZ+1e-10))
            deact = s*(1-s)

        return deact
    
    def linear_backward(self,dA,n_layer):
        '''
        Calculates linear backward propragation for layer denoted by n_layer

        :param dA: Derivative of cost w.r.t this layer
        :param n_layer: layer number
        :return : dZ,dW,db,dA_prev
        '''
        batch_size = dA.shape[1]
        current_cache = self.cache[n_layer-1]
        linear_cache = current_cache[0]
        A_prev,W,b = linear_cache

        dZ = dA*self.deactivate(dA,n_layer)
        dW = (1/batch_size)*dZ.dot(A_prev.T) + (self.lamb/batch_size)*self.parameters['W'+str(n_layer)]
        db = (1/batch_size)*np.sum(dZ,keepdims=True,axis=1)
        dA_prev = W.T.dot(dZ)

        assert(dA_prev.shape == A_prev.shape)
        assert(dW.shape == W.shape)
        assert(db.shape == b.shape)
        
        return dW,db,dA_prev
        
        

    def backward(self,prediction,mappings):
        '''
        Backward propagates through the network and stores useful calculations

        :param prediction: Output of neural net
        :param mapping: Correct output of the function
        :return : None
        '''
        layer_num = len(self.cache)
        doutput = self.output_backward(prediction,mappings)

        self.grads['dW'+str(layer_num)],self.grads['db'+str(layer_num)],self.grads['dA'+str(layer_num-1)] = self.linear_backward(doutput,layer_num)

        for l in reversed(range(layer_num-1)):

            if self.layer_type[l] == 'fc':
                dW,db,dA_prev = self.linear_backward(self.grads['dA'+str(l+1)],l+1)

            self.grads['dW'+str(l+1)] = dW
            self.grads['db'+str(l+1)] = db
            self.grads['dA'+str(l)] = dA_prev
    
    @staticmethod
    def zero_pad(imgData,pad):
        '''
        Provides zero padding to the multi channel image data provided
        :param imgData: image data to pad
        :param pad    : amount of padding per layer

        :return : image with desired padding
        '''
        X = np.pad(imgData,((0,0),(pad,pad),(pad,pad),(0,0)),'constant',constant_values = 0)
        return X

    
   
    def conv_single(self,a_prev_slice,W,b):
        '''
        Apply convolution using W and b as filter on the activation slice of the previous layer

        :param a_prev_slice: a slice of previous activated layer
        :param W           : Filter
        :param b           : bais
        :return Z: scalar value resultant of the convolution
        '''
        Z  = np.multiply(a_prev_slice,W)
        Z = np.sum(Z)
        Z = Z + float(b) #to convert the value to float from matrix type
        return Z
    
    def conv_forward(self,A_prev,W,b,hyper_param):
        '''
        Implements forward pass of convolutional layer.
        
        :param A_prev:activations of previous layer
        :param W: Filter
        :param b: bias
        :param hyper_param  : list of hyperparameters, stride and padding

        :return: Z,cache
        '''
        m,h_prev,w_prev,nc_prev = A_prev.shape
        f,f,nc_prev,nc = W.shape
        stride,pad = hyper_param
        #comupte the dimensions of the result using convolution formula => w/h = (w/h(prev) -f +2*pad)/stride +1
        n_h = int(np.floor((h_prev-f+2*pad)/2)) +1
        n_w = int(np.floor((w_prev-f+2*pad)/2)) +1
        
        Z = np.zeros((m,n_h,n_w,nc))
        A_prev_pad = self.zero_pad(A_prev,pad)
        for i in range(m):
            prev_pad = A_prev_pad[i]

            for h in range(n_h):
                for w in range(n_w):
                    for c in range(nc):
                        vstart = h*stride
                        vend = vstart + f
                        hstart = w*stride
                        hend = hstart+f 

                        prev_slice = prev_pad[vstart:vend,hstart:hend,:]
                        Z[i,h,w,c] = self.conv_single(prev_slice,W[:,:,:,c],b[:,:,:,c])
        
        cache = (A_prev,W,b,hyper_param)

        return Z,cache


    def __str__(self):
        '''
        :Return: the network architecture and connectivity
        '''
        net_string = ""
        for params in range(int(len(self.parameters)/2)):
            weight = self.parameters['W'+str(params+1)]
            net_string = net_string + " -> Linear(" + str(weight.shape[1]) +" , " + str(weight.shape[0]) + ")"
            if self.activations[params] != None:
                net_string = net_string + " -> " +  self.activations[params]
        return net_string

            
net = nn([100],['relu'])
np.random.seed(1)
A_prev = np.random.randn(10,4,4,3)
W = np.random.randn(2,2,3,8)
b = np.random.randn(1,1,1,8)
hparameters = {"pad" : 2,
               "stride": 2}

Z, cache_conv = net.conv_forward(A_prev, W, b, hyper_param=[2,2])
print("Z's mean =", np.mean(Z))
print("Z[3,2,1] =", Z[3,2,1])
print("cache_conv[0][1][2][3] =", cache_conv[0][1][2][3])
