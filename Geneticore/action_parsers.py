from numpy import exp

class DiscreteAction:
  def __init__(self, bins):
    self.bins = bins
    
  def softmax(self, activations):
    e = exp(activations)
    return e / e.sum() 

  def find_output(self, activations):
    #Take softmax activations and get the highest one
    max = [0, 0]
    for i in range(self.bins):
      if activations[i] > max[0]:
        max[0] = activations[i]
        max[1] = i

    return max[1]

  def parse(self, activations):
    s_f_activations = self.softmax(activations)
    output = self.find_output(s_f_activations)

    return output

class ContiniousAction:
  def __init__(self, range=1, graph_fun=None):
    self.range = range
    self.graph_fun = graph_fun

  def parse(self, activation):
    activation = activation[0]
    output = self.graph_fun(activation, self.range)

    return output