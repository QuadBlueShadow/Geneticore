import numpy as np

class Loop:
  def __init__(self, alg, f_c, env=None):
    self.alg = alg
    self.f_c = f_c
    self.env = env
    
  #Use this when your envoirnment does not have a truncated variable
  def training_loop(self):
    try:
      while True:
        #Setup our nets and rewards
        nets = self.alg.give_nets()
        rewards = []
        total_rewards = np.array([])

        for i in range(len(nets)):
          net = nets[i]
          done = False

          obs = self.env.reset()

          net_rewards = np.array([])

          #Standard gymnasium stuff
          while not done:
            action = net.run(obs)
            obs, reward, done, info = self.env.step(action)

            net_rewards = np.append(net_rewards, reward)
            total_rewards = np.append(total_rewards, reward)
        
            obs = obs.flatten()

          rewards.append(net_rewards)

        self.f_c.calc(np.asarray(rewards)) #Track fitness

        fitnesses = self.f_c.get_fitnesses() #Do this when you need to make a new generation

        self.alg.step(total_rewards, fitnesses) #Rewards for that step, fitnesses only need to be inputted on the step optimization happens

        fitnesses = self.f_c.clear() #Make sure to clear the past generation's fitnesses
        
    except KeyboardInterrupt:
      #Easy way of saving models
      self.alg.save_best_net()
      print("Exiting training")
  
  #Use this when you have a truncated variable in your gym
  def training_loop_t(self):
    try:
      while True:
        #Setup our nets and rewards
        nets = self.alg.give_nets()
        rewards = []
        total_rewards = []

        for i in range(len(nets)):
          net = nets[i]
          done = False
          truncated = False

          obs, info = self.env.reset()

          net_rewards = []

          #Standard gymnasium stuff
          while not done and not truncated:
            action = net.run(obs)
            obs, reward, done, truncated, info = self.env.step(action)

            net_rewards = np.append(net_rewards, reward)
            total_rewards = np.append(total_rewards, reward)
        
            obs = obs.flatten()

          rewards = np.append(rewards, net_rewards)

        self.f_c.calc(rewards) #Track fitness

        fitnesses = self.f_c.get_fitnesses() #Do this when you need to make a new generation

        self.alg.step(total_rewards, fitnesses) #Rewards for that step, fitnesses only need to be inputted on the step optimization happens

        fitnesses = self.f_c.clear() #Make sure to clear the past generation's fitnesses
        
    except KeyboardInterrupt:
      #Easy way of saving models
      self.alg.save_best_net()
      print("Exiting training")