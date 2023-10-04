import fc_layer as fc
import tqdm
class NN:
    def __init__(self,name="Model"):
        self.name = name
        self.layers= []

    def __repr__(self):
        return f'''
---------------------
Model name : {self.name}
Model layers : {self.layers}
---------------------
    '''
    def add(self,layer):
        self.layers.append(layer)


    def forward(self, inputs):
        output = inputs
        for layer in tqdm.tqdm(self.layers):
            output = layer.feed_forward(output) 

        return output
