import random as r
from numpy import exp
import numpy as np
def leaky_relu(x):
  if x > 0:
    return x
  else:
    return x * 0.01

def create_nueron(n_layer):
  weights = np.zeros(n_layer)
  biases = np.zeros(n_layer)

  #Go through each weight
  for i in range(len(weights)):
    #Set the modifier, are we adding or subtracting from this weight
    modifier = 0
    if r.randint(0, 1) == 0:
      modifier = r.random() / 5
    else:
      modifier = -1 * (r.random() / 5)

    #Change weight
    weights[i] += modifier

    biases[i] = r.random()*r.randint(-100, 100)

  return weights, biases

def r_adjust_n_weights(weights):
  #Go through each weight
  for i in range(len(weights)):
    #Set the modifier, are we adding or subtracting from this weight
    modifier = 0
    if r.randint(0, 1) == 0:
      modifier = r.random() / 5
    else:
      modifier = -1 * (r.random() / 5)

    #Change weight
    weights[i] += modifier

  return weights

class Layer:
  def __init__(self, num, n_layer):
    self.weights = []
    self.biases = []

    self.n_layer = n_layer

    for i in range(num):
      weight, bias = create_nueron(n_layer)
      self.weights.append(weight)
      self.biases.append(bias)

    self.weights = np.asarray(self.weights)
    self.biases = np.asarray(self.biases)

  def run(self, inputs=[]):
    full_output = np.zeros(self.n_layer)
    outputs = []

    #Get all of the outputs of one neuron
    for i in range(len(inputs)):
      outputs.append(inputs[i] * self.weights[i] + self.biases[i])

    outputs = np.asarray(outputs)

    #In context of the next layer
    #Get all the outputs going to the neruon 0, 1, 2, 3 ...
    for i in range(len(outputs)):
      output = outputs[i]
      for o in range(len(output)):
        full_output[o] += output[o]

    return full_output

  def random_adjust_n(self):
    #Randomly adjust neuron weights
    for weights in self.weights:
      r_adjust_n_weights(weights)

    for bias in self.biases:
      r_adjust_n_weights(bias)

class Net:
  def __init__(self, net_arr=[1, 64, 64, 1], act_parser=None):
    self.layers = []
    self.act_parser = act_parser

    #Make the layer of the neural net
    for i in range(len(net_arr)):
      self.layers.append(Layer(net_arr[i], net_arr[i + 1]))
      if i + 2 == len(net_arr):
        break

    self.layers = np.asarray(self.layers)

  def find_output(self, activations):
    #Take softmax activations and get the highest one
    max = np.array([0, 0])
    for i in range(len(activations)):
      if activations[i] > max[0]:
        max[0] = activations[i]
        max[1] = i

    return max[1]

  def run(self, obs):
    #Store previous layer outputs
    p_l = None
    
    #Run through each layer
    for i in range(len(self.layers)):
      layer = self.layers[i]

      #Use the obs as input for the first layer
      if i == 0:
        p_l = layer.run(obs)
      else:
        p_l = layer.run(p_l)

    #Grab the net's output
    out = self.act_parser.parse(p_l)

    return out

  def randomly_adjust_net(self):
    #Run through each net and randomly set neuron wieghts
    for layer in self.layers:
      layer.random_adjust_n()