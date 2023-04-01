class Loop:
  def __init__(self, alg, f_c, steps=1_000):
    self.alg = alg
    self.f_c = f_c
    self.steps = steps
    self.c_step = 0
    
  def training_loop(self):
    rewards = [1] #One net, one reward value, Disclaimer: DONT USE ONE NET
    self.f_c.calc(rewards) #Track fitness
    fitnesses = []
    
    if self.c_step >= self.steps:
      fitnesses = self.f_c.get_fitnesses() #Do this when you need to make a new generation
    
    self.alg.step(rewards, fitnesses) #Rewards for that step, fitnesses only needs to be inputted on the step optimization happens

    if self.c_step >= self.steps:
      fitnesses = self.f_c.clear() #Make sure to clear the past generation's fitnesses
