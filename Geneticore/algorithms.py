import pickle

class CRGeneticAlg:
  def __init__(self,  base_net=None, episodes_per_gen=1, num_nets=5, t_episodes=0, max_fitness=-10000):
    #Know when to optimize things
    self.episodes_per_gen = episodes_per_gen
    self.c_episode = 0
    
    #Our base variables
    self.base_net = base_net

    #Net stuff
    self.best_net = base_net
    self.nets = []
    self.num = num_nets
    self.abs_best = base_net

    #Cool info
    self.generation = 1
    self.max_fitness = max_fitness
    self.avg_reward = 0
    self.t_episodes = t_episodes

  def make_nets(self):
    #Create a net from the base and make small random adjustments to the weights
    for i in range(self.num):
      if i < self.num/10:
        new_net = self.abs_best
        new_net.randomly_adjust_net()
        self.nets.append(new_net)
      else:
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
    max_fitness = 0

    for net, fitness in info:
      if fitness > max_fitness:
        max_fitness = fitness
        self.best_net = net

      if fitness >= self.max_fitness:
        self.max_fitness = fitness
        self.abs_best = net

    return max_fitness

  def calculate_reward(self, rewards):
    for reward in rewards:
      if self.avg_reward == 0:
        self.avg_reward = reward
      else:
        self.avg_reward = (reward + self.avg_reward)/2
  
  def step(self, rewards=[], fitnesses=[]):
    if self.c_episode >= self.episodes_per_gen:
      #Reset things and make a new set of nets
      self.c_episode = 0
      m_f = self.choose_best_net(fitnesses)
      self.clear_nets()
      self.make_nets()
      self.print_stats(m_f)
      self.generation += 1
    else:
      #Track the average reward of the generation
      self.calculate_reward(rewards)
      self.c_episode += 1
      self.t_episodes += 1
      return self.nets
    
  def save_best_net(self):
    print("MAX FITNESS:", self.max_fitness)
    pickle_out = open(f"model_{self.t_episodes}_episodes.pickle","wb")
    pickle.dump([self.abs_best, self.t_episodes, self.max_fitness], pickle_out)
    pickle_out.close()

  def print_stats(self, max_g_fitness):
    print("Generation:", self.generation)
    print("avg_reward:", self.avg_reward)
    print("max_fitness:", self.max_fitness)
    print("max_generation_fitness:", max_g_fitness)
    print("num_nets:", self.num)
    print(" ")