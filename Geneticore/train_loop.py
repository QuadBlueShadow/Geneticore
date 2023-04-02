import numpy as np

class Loop:
  def __init__(self, alg, f_c, steps=1_000, env=None):
    self.alg = alg
    self.f_c = f_c
    self.steps = steps
    self.c_step = 0
    self.env = env
    
  def training_loop(self, rendering=True):
    try:
      while True:
        if self.c_step >= self.steps+1:
          self.c_step = 0

        nets = self.alg.give_nets()
        rewards = []
        total_rewards = []

        for i in range(len(nets)):
          net = nets[i]
          done = False

          obs, info = self.env.reset()

          net_rewards = []

          while not done:
            action = net.run(obs)
            obs, reward, done, truncated, info = self.env.step(action)

            net_rewards.append(reward)
            total_rewards.append(reward)
        
            obs = obs.flatten()

            if rendering:
              self.env.render()

          rewards.append(net_rewards)

        self.f_c.calc(rewards) #Track fitness

        fitnesses = self.f_c.get_fitnesses() #Do this when you need to make a new generation

        self.alg.step(total_rewards, fitnesses) #Rewards for that step, fitnesses only need to be inputted on the step optimization happens

        fitnesses = self.f_c.clear() #Make sure to clear the past generation's fitnesses

        self.c_step += 1
        
    except KeyboardInterrupt:
      self.alg.save_best_net()
      print("Exiting training")