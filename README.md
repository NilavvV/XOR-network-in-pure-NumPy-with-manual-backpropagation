#XOR LOGIC USING NN & BACKPROP

So, what I did here is make a very simple neural network model
A 2-2-1 neural network (2 inputs, 2 hidden neurons, 1 output) that learns XOR,
with forward pass and backpropagation written by hand. No ML libraries.

to learn and implement a logic gate xor , although we can implement any gate I decided to choose an xor gate because its not linear and it more fun 

## How it works
- Forward pass: (sigmoid at hidden layer and output, in your own words)
- Loss: (which loss, and the formula)
- Backpropagation: (how the error reaches the hidden layer, mention the chain rule)
- Updates: (gradient descent, learning rate, epochs)


also this is taught in our class in theory , I decide to tweak and play with it by making it into a model , it was fun implementing what idea i had about neural network from my class 

the python i wrote is very beginner , i am still learning and promise to work on improving it 

##Results:
The result that I obtained on training the model
Training the neural network on XOR gate:

Testing predictions after training:
Input: (0, 0) -> Predicted: 0.0050
Input: (0, 1) -> Predicted: 0.9943
Input: (1, 0) -> Predicted: 0.9953
Input: (1, 1) -> Predicted: 0.0044

## Known limitations
- Initialization is random, so some runs can get stuck. (Seed used: ___)
- Plain Python loops, not vectorized. The next repo (MNIST) fixes that.

## Run it
python nn_logic_XOR.py
