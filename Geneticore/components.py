import random as r
from numpy import exp

def softmax(activations):
  e = exp(activations)
  return e / e.sum()

def leaky_relu(x):
  if x > 0:
    return x
  else:
    return x * 0.01

class Neuron:
  def __init__(self, num):
    self.weights = []

    for i in range(num):
      self.weights.append(r.random())

  def run(self, input):
    outputs = []

    #Leaky relu activation for the input, change this if you want or whatever
    a_in = leaky_relu(input)

    #Get each individual weights output
    for i in range(len(self.weights)):
      outputs.append(a_in * self.weights[i])

    return outputs

  def random_adjust_w(self):
    #Go through each weight
    for i in range(len(self.weights)):
      #Set the modifier, are we adding or subtracting from this weight
      modifier = 0
      if r.randint(0, 1) == 0:
        modifier = r.random() / 5
      else:
        modifier = -1 * (r.random() / 5)

      #Change weight
      self.weights[i] += modifier

class Layer:
  def __init__(self, num, n_layer):
    self.neurons = []
    self.n_layer = n_layer
    self.full_output = []

    #Create the whole layer
    for i in range(n_layer):
      self.full_output.append(0)

    for i in range(num):
      new_neuron = Neuron(n_layer)
      self.neurons.append(new_neuron)

  def run(self, inputs=[]):
    outputs = []

    #Use this to store the full output going out to each neuron in the next layer
    for i in range(len(self.full_output)):
      self.full_output[i] = 0

    #Get all of the outputs of one neuron
    for i in range(len(inputs)):
      outputs.append(self.neurons[i].run(inputs[i]))

    #In context of the next layer
    #Get all the outputs going to the neruon 0, 1, 2, 3 ...
    for i in range(len(outputs)):
      output = outputs[i]
      for o in range(len(output)):
        self.full_output[o] += output[o]

    return self.full_output

  def random_adjust_n(self):
    #Randomly adjust neuron weights
    for neuron in self.neurons:
      neuron.random_adjust_w()

class Net:
  def __init__(self, net_arr=[1, 64, 64, 1]):
    self.layers = []

    #Make the layer of the neural net
    for i in range(len(net_arr)):
      self.layers.append(Layer(net_arr[i], net_arr[i + 1]))
      if i + 2 == len(net_arr):
        break

  def find_output(self, activations):
    #Take softmax activations and get the highest one
    max = [0, 0]
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
    activations = softmax(p_l)
    out = self.find_output(activations)

    return out

  def randomly_adjust_net(self):
    #Run through each net and randomly set neuron wieghts
    for layer in self.layers:
      layer.random_adjust_n()