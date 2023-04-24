from Geneticore.components import Net
from Geneticore.algorithms import CRGeneticAlg
from Geneticore.fitness_calculator import FC
from Geneticore.train_loop import Loop
from Geneticore.loading import load_model
from Geneticore.action_parsers import DiscreteAction

import gymnasium as gym

N_NETS = 100
t_episodes = 0
max_fitness = -1000

act_parser = DiscreteAction(2) #Make a discrete action for our env
base_net = Net(net_arr=[4, 20, 20, 2], act_parser=act_parser) #Create a base net to work with
#base_net, t_episodes, max_fitness = load_model("model_34_episodes.pickle")

alg = CRGeneticAlg(base_net=base_net, episodes_per_gen=1, num_nets=N_NETS, t_episodes=t_episodes, max_fitness=max_fitness) #Create the algorithm and put the base net in it
f_c = FC(num_nets=N_NETS) #Not needed, just makes it easier to track fitness values for later on

env = gym.make("CartPole-v1")

alg.make_nets() #The first nets we are going to use

t_loop = Loop(alg=alg, f_c=f_c, env=env)
t_loop.training_loop()
