import fc_layer as fc
import numpy as np


dense_1 = fc.fc_layer(np.array([1,2,3]),3,5,"Dense_1","relu")
print(dense_1.weights_matrix)
activations = dense_1.feed_forward()

print(dense_1.weights_matrix)
print(activations)
