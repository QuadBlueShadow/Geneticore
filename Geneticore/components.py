import random as r
import numpy as np

def leaky_relu(x):
  return x if x>0 else x*0.01

class Layer:
  def __init__(self, prev_layer_len, layer_len, act_fun):
    self.act_fun = act_fun
    #Initialize weights and biases, all with a random value from -1 to 1
    self.weights = np.array([[r.uniform(-1, 1) for i in range(prev_layer_len)] for j in range(layer_len)])
    self.biases = np.array([r.uniform(-1, 1) for i in range(layer_len)])

  def run(self, inputs: np.ndarray):
    #Some simple math that gets this done quicker than the way Quad had (no offense Quad)
    #ONE LINE YESSSSSSSSSSSS --CTA
    return [self.act_fun(i) for i in self.weights.dot(inputs) + self.biases]

  def random_adjust_n(self):
    #Randomly adjust weights and biases by any value from -0.2 to +0.2
    for i in range(len(self.weights)):
      for j in range(len(self.weights[i])):
        self.weights[i][j] += r.uniform(-0.2, 0.2)
      self.biases[i] += r.uniform(-0.2, 0.2)

class Net:

  def __init__(self, net_arr=[1, 64, 64, 1], act_parser=None, act_fun=leaky_relu):
    self.layers = []
    self.act_parser = act_parser
    #Make the layer of the neural net
    for i in range(len(net_arr)-1):
      self.layers.append(Layer(net_arr[i], net_arr[i + 1], act_fun))

  def find_output(self, activations):
    #Take softmax activations and get the highest one
    #One line funny
    return list(activations).index(max(activations))

  def run(self, input):
    #Run through each layer
    for i in self.layers:
      input = i.run(input)
    #Act parser woooo
    return self.act_parser.parse(input)

  def randomly_adjust_net(self):
    #Run through each net and randomly set neuron wieghts
    for layer in self.layers:
      layer.random_adjust_n()
