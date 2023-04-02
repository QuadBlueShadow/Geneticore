from Geneticore.components import Net
from Geneticore.main_algorithm import CRGeneticAlg
from Geneticore.fitness_calculator import FC
from Geneticore.train_loop import Loop
from Geneticore.loading import load_model

import gymnasium as gym

N_NETS = 30

base_net = Net(net_arr=[8, 10, 10, 3]) #Create a base net to work with
#base_net = load_model()
alg = CRGeneticAlg(base_net=base_net, steps=5, num_nets=N_NETS) #Create the algorithm and put the base net in it
f_c = FC(num_nets=N_NETS, divisor=10) #Not needed, just makes it easier to track fitness values for later on

env = gym.make("LunarLander-v2")

alg.make_nets() #The first nets we are going to use

t_loop = Loop(alg=alg, f_c=f_c, steps=10, env=env)
t_loop.training_loop(rendering=False)