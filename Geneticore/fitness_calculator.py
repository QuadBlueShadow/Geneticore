import numpy as np

#Easily keep track of fitnesses to use in the algorithm
class FC:
  def __init__(self):
    self.fitnesses = []

  def calc(self, rewards=[]):
    self.fitnesses = self.fitnesses + [sum(cr) for cr in rewards]

  def clear(self):
    self.fitnesses = []
  
  def get_fitnesses(self):
    return self.fitnesses