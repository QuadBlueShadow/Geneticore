#Easily keep track of fitnesses to use in the algorithm
class FC:
  def __init__(self, num_nets=1, divisor=10):
    self.fitnesses = []
    self.divisor = divisor

    for i in range(num_nets):
      self.fitnesses.append(0)

  def calc(self, rewards=[]):
    for i in range(len(rewards)):
      curr_rewards = rewards[i]

      for x in range(len(curr_rewards)):
        self.fitnesses[i] += curr_rewards[x]/self.divisor

  def clear(self):
    for i in range(len(self.fitnesses)):
      self.fitnesses[i] = 0
  
  def get_fitnesses(self):
    return self.fitnesses