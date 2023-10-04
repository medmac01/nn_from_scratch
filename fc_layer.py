import numpy as np

class fc_layer:
    def __init__(self,inputs,in_size,out_size,name="Dense",activation="relu"):
        self.weights_matrix = np.zeros(shape=(in_size,out_size), dtype=np.float16)
        self.layer_name = name
        self.activation = activation
        self.shape = (in_size,out_size)
        self.inputs = inputs

        assert inputs.shape[0] == int(in_size), "Inputs and out_size do not match"

    def init_params(self):
        self.weights_matrix = np.random.uniform(size=self.shape)


    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def relu(self,x):
        x[x < 0] = 0
        return x

    def feed_forward(self):
        self.init_params()
        weighted_sum = np.matmul(self.weights_matrix.T,self.inputs.T)
        activations = self.sigmoid(weighted_sum) if self.activation == 'sigmoid' else self.relu(weighted_sum)
        return activations
