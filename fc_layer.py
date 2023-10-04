import numpy as np

class fc_layer:
    def __init__(self,in_size,out_size,name="Dense",activation="relu"):
        self.weights_matrix = np.zeros(shape=(in_size,out_size), dtype=np.float16)
        self.layer_name = name
        self.activation = activation
        self.shape = (in_size,out_size)


    def __repr__(self):
        info = f'''
        Layer name : {self.layer_name}
        Size : {self.shape[1]}
        Activation : {self.activation}
        '''
        return info

    def init_params(self):
        self.weights_matrix = np.random.uniform(size=self.shape)


    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def relu(self,x):
        x[x < 0] = 0
        return x

    def feed_forward(self,inputs):
        self.init_params()
        assert inputs.shape[0] == int(self.shape[0]), "Inputs and out_size do not match"
        weighted_sum = np.matmul(self.weights_matrix.T,inputs.T)
        activations = self.sigmoid(weighted_sum) if self.activation == 'sigmoid' else self.relu(weighted_sum)
        return activations
