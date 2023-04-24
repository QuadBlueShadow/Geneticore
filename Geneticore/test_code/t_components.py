import random as r

def leaky_relu(x):
  if x > 0:
    return x
  else:
    return x * 0.01

class Neuron:
  def __init__(self, num):
    self.weights = []
    self.adj = []

    for i in range(num):
      self.weights.append(r.random())
      self.adj.append(0)

  def get_weights(self):
    return self.weights

  def get_adj(self):
    return tuple(self.adj)

  def run(self, input):
    outputs = []

    #Leaky relu activation for the input, change this if you want or whatever
    a_in = leaky_relu(input)

    #Get each individual weights output
    for i in range(len(self.weights)):
      outputs.append(a_in * self.weights[i])

    return outputs

  def random_adjust_w(self, biases):
    #Go through each weight
    for i in range(len(self.weights)):
      #Set the modifier, are we adding or subtracting from this weight
      modifier = 0
      if r.randint(0, 1) == 0:
        modifier = r.random()/2
      else:
        modifier = -1 * (r.random()/2)

      #Change weight
      adj = modifier + (biases[i]/4)
      self.weights[i] += adj
      self.adj[i] = adj

class Layer:
  def __init__(self, num, n_layer):
    self.neurons = []
    self.n_layer = n_layer
    self.full_output = []

    self.adj = []

    #Create the whole layer
    for i in range(n_layer):
      self.full_output.append(0)

    for i in range(num):
      new_neuron = Neuron(n_layer)
      adj = new_neuron.get_adj()
      self.neurons.append(new_neuron)
      self.adj.append(adj)

  def get_weights(self):
    l_weights = []
    for neuron in self.neurons:
      l_weights.append(neuron.get_weights())

    return l_weights

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
      for o in output:
        self.full_output += o

    return self.full_output

  def random_adjust_n(self, layer_biases):
    #Randomly adjust neuron weights
    for i in range(len(self.neurons)):
      self.neurons[i].random_adjust_w(layer_biases[i])
      self.adj[i] = self.neurons[i].get_adj()

  def get_adj_l(self):
    return tuple(self.adj)

class Net:
  def __init__(self, net_arr=[1, 64, 64, 1], act_parser=None):
    self.layers = []
    self.adj = []

    self.act_parser = act_parser

    #Make the layer of the neural net
    for i in range(len(net_arr)):
      new_layer = Layer(net_arr[i], net_arr[i + 1])
      
      self.adj.append(new_layer.get_adj_l())
      
      self.layers.append(new_layer)
      if i + 2 == len(net_arr):
        break

  def get_weights(self):
    full_weights = []
    for layer in self.layers:
      full_weights.append(layer.get_weights())

    return full_weights

  def run(self, obs):
    #Store previous layer outputs
    p_l = None
    first = True

    #Run through each layer
    for layer in self.layers:
      #Use the obs as input for the first layer
      if first:
        p_l = layer.run(obs)
        first = False
      else:
        p_l = layer.run(p_l)

    #Grab the net's output
    out = self.act_parser.parse(p_l)

    return out

  def randomly_adjust_net(self, biases):
    #Run through each net and randomly set neuron wieghts
    for i in range(len(self.layers)):
      self.layers[i].random_adjust_n(biases[i])
      self.adj[i] = self.layers[i].get_adj_l()

  def get_adj_n(self):
    return tuple(self.adj)