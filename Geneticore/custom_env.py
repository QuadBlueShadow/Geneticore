#This imitates a gymnasium enviornment
class CustomEnv:
  def __init__(self, base_obs=[], action_space = [1, [0, 1]], reward_fun=None, action_fun=None, state_setter=None):
    self.action_space = action_space #The action type and length: 1 action, 2 values: 0, 1
    self.obs = base_obs #The obs values you will use in your env. This sets the first obs.
    self.reward_fun = reward_fun #Reward function
    self.action_fun = action_fun #This is how the agent will interact with the obs
    self.state_setter = state_setter

  #This is how you continue with actions in your env
  def step(self, action):
    #Things you might want to return
    done, truncated = self.action_fun(self.obs)
    return self.obs, self.reward_fun(self.obs), done, truncated #Info is left out of our return, feel free to add anything else you need

  def reset(self):
    #Reset env
    self.obs = self.state_setter(self.obs)