import numpy as np

class DiscreteAction:
  def __init__(self, bins):
    self.bins = bins

  def find_output(self, activations):
    #I made it one line lmao --CTA
    return list(activations).index(max(activations))

  def parse_act(self, activations):
    output = self.find_output(activations)
    return output
  
  def return_act_space(self):
      return self.bins
  
  def parse(self, activations):
    return self.parse_act(activations)

class ContiniousAction:
  def __init__(self, range=1, graph_fun=None):
    self.range = range
    self.graph_fun = graph_fun

  def parse(self, activation):
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

    def parse(self, activations) -> np.ndarray:
        action = self.parse_act(activations)
        return self._lookup_table[action]