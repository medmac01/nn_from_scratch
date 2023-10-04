import fc_layer as fc
class NN:
    def __init__(self,name="Model"):
        self.name = name
        self.layers= []
    def add(self,layer):
        self.layers.append(layer)

    def forward(self, inputs):
        output = inputs
        for layer in self.layers:
            output = layer.feed_forward(output) 

        return output
