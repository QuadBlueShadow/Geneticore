import pickle

#CR stands for Completely Random adjustments
class CRGeneticAlg:
  def __init__(self, base_net=None, steps=1_000, num_nets=5):
    #Know when to optimize things
    self.steps = steps
    self.c_step = 0
    
    #Keep this here just in case
    self.base_net = base_net

    #Net stuff
    self.best_net = base_net
    self.nets = []
    self.num = num_nets

    #Cool info
    self.generation = 1
    self.max_fitness = 0
    self.avg_reward = 0

  def make_nets(self):
    #Create a net from the base and make small random adjustments to the weights
    for i in range(self.num):
      new_net = self.best_net
      new_net.randomly_adjust_net()
      self.nets.append(new_net)

    return self.nets

  def clear_nets(self, best_net=False):
    #Clear our current nets
    self.nets = []
    #Clear best net for a full restart or something
    if best_net:
      self.best_net = self.base_net

  def give_nets(self):
    return self.nets

  def choose_best_net(self, fitnesses=[]):
    #Put nets and fitness into one array
    info = []
    for i in range(len(self.nets)):
      info.append([self.nets[i], fitnesses[i]])

    #Find best net
    #max_fitness = 0
    
    for net, fitness in info:
      if fitness > self.max_fitness:
        self.max_fitness = fitness
        self.best_net = net

      #Track the absolute best fitness
      # if fitness > self.max_fitness:
      #   self.max_fitness = fitness

  def calculate_reward(self, rewards):
    for reward in rewards:
      if self.avg_reward == 0:
        self.avg_reward = reward
      else:
        self.avg_reward = (reward + self.avg_reward)/2
  
  def step(self, rewards=[], fitnesses=[]):
    if self.c_step >= self.steps:
      #Reset things and make a new set of nets
      self.c_step = 0
      self.choose_best_net(fitnesses)
      self.clear_nets()
      self.make_nets()
      self.print_stats()
      self.generation += 1
    else:
      #Track the average reward of the generation
      self.calculate_reward(rewards)
      self.c_step += 1
      return self.nets
    
  def save_best_net(self):
    pickle_out = open("model.pickle","wb")
    pickle.dump(self.best_net, pickle_out)
    pickle_out.close()

  def print_stats(self):
    print("Generation:", self.generation)
    print("avg_reward:", self.avg_reward)
    print("max_fitness:", self.max_fitness)
    print("num_nets:", self.num)
    print(" ")