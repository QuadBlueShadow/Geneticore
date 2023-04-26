import numpy as np

#Easily keep track of fitnesses to use in the algorithm
class FC:
  def __init__(self):
    self.fitnesses = []

  def calc(self, rewards=[]):
    for cr in rewards:
      self.fitnesses.append(sum(cr))

    self.fitnesses = np.asarray(self.fitnesses)

  def clear(self):
    self.fitnesses = []
  
  def get_fitnesses(self):
    return self.fitnesses