import numpy as np 
inputs = [(0,0),(0,1),(1,0),(1,1)]
weights = np.random.randn(2,4)*0.5
lr = 0.5
epochs = 100000

#defining the sigmoid and sigmoid derivative 
def sigmoid (a):
    y = 1 + np.exp(-a)
    return 1/y
def sigmoid_derivative(a):
    return a*(1-a)
def neural_network(inputs,weights, expected_output, lr , epochs):
    
    w11, w12 , b1, w_out1 = weights[0]
    w21, w22 , b2 , w_out2 = weights[1]
    b = np.random.rand()
    for epoch_idx in range(epochs):
        total_loss = 0
        for x,y_true in zip(inputs,expected_output):
            #----forwardpass------
            # --- Forward Pass ---
            # Hidden layer calculations
            z_1 = x[0] * w11 + x[1] * w21 - b1
            z_2 = x[0] * w12 + x[1] * w22 - b2
            h_1 = sigmoid(z_1)
            h_2 = sigmoid(z_2)

            # Output layer calculations
            z = h_1 * w_out1 + h_2 * w_out2 - b
            y_pred = sigmoid(z)

            #loss_calculation 
            loss = 0.5 * (y_true - y_pred) ** 2
            total_loss += loss

            #backpropagation now at outputlayer

            error_out = y_pred - y_true
            delta_out = error_out * sigmoid_derivative(y_pred)

            #at hidden layer 
            delta_h1 = delta_out * sigmoid_derivative(h_1) * w_out1
            delta_h2 = delta_out * sigmoid_derivative(h_2) * w_out2

            #Weight updation Gradient descent 
            #----------output_layer-----------#
            w_out1 -= lr * delta_out * h_1
            w_out2 -= lr * delta_out * h_2
            b += lr * delta_out

            # Hidden layer 1 weights and bias updates
            w11 -= lr * delta_h1 * x[0]
            w21 -= lr * delta_h1 * x[1]
            b1 += lr * delta_h1

            # Hidden layer 2 weights and bias updates
            w12 -= lr * delta_h2 * x[0]
            w22 -= lr * delta_h2 * x[1]
            b2 += lr * delta_h2

        #if (epoch_idx + 1) % 100 == 0 or epoch_idx == 0:
            #print(f"Epoch {epoch_idx + 1}/{epochs} - Loss: {total_loss / len(inputs):.6f}")

    #update weights and biases
    weights[0] = [w11,w21,b1,w_out1]
    weights[1] = [w12,w22,b2,w_out2]
    return weights , b
y_xor = [0, 1, 1, 0]

print("Training the neural network on XOR gate:")
trained_weights, final_bias = neural_network(inputs, weights, y_xor, lr, epochs)

print("\nTesting predictions after training:")
for x in inputs:
    z_1 = x[0] * trained_weights[0][0] + x[1] * trained_weights[0][1] - trained_weights[0][2]
    z_2 = x[0] * trained_weights[1][0] + x[1] * trained_weights[1][1] - trained_weights[1][2]
    h_1 = sigmoid(z_1)
    h_2 = sigmoid(z_2)
    z = h_1 * trained_weights[0][3] + h_2 * trained_weights[1][3] - final_bias
    y = sigmoid(z)
    print(f"Input: {x} -> Predicted: {y:.4f}")




            


