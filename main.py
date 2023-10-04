import fc_layer as fc
import numpy as np
import nn

dense_1 = fc.fc_layer(3,5,"Dense_1","relu")
print(dense_1)
#print(dense_1.weights_matrix)
activations = dense_1.feed_forward(np.array([1,2,4.1]))

#print(dense_1.weights_matrix)
#print(activations)


model = nn.NN()
model.add(fc.fc_layer(in_size=784, out_size=512, name="dense1", activation="relu"))
model.add(fc.fc_layer(in_size=512, out_size=256, name="dense2", activation="relu"))
model.add(fc.fc_layer(in_size=256, out_size=10, name="dense3", activation="relu"))

inputs = np.random.random(size=784)
#inputs = np.array([1, 2, 3, 4, 5], dtype=np.float16)

# Forward pass through the network
output = model.forward(inputs)

print(output)
