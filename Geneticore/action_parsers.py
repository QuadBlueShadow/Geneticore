import numpy as np
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
  
  def get_action_space(self):
      return self.bins
  
  def parse_actions(self, activations):
    return self.parse(activations)

class ContiniousAction:
  def __init__(self, range=1, graph_fun=None):
    self.range = range
    self.graph_fun = graph_fun

  def parse_actions(self, activation):
    activation = activation[0]
    output = self.graph_fun(activation, self.range)

    return output
  
#Make a lookup table of all actions
#Works as a multidiscrete action
class LookupAction(DiscreteAction):
    def __init__(self, bins=[(-1, 0, 1)] * 5):
        self.act_bins = bins

        self._lookup_table = self.make_lookup_table(self.act_bins)

        self.bins = len(self._lookup_table)

    def make_lookup_table(self, bins):
        actions = []
        #Write your lookup algorithm here
        actions = np.array(actions)
        return actions

    def parse_actions(self, activations) -> np.ndarray:
        action = self.parse(activations)
        return self._lookup_table[action]