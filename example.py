from Geneticore.components import Net
from Geneticore.main_algorithm import CRGeneticAlg
from Geneticore.fitness_calculator import FC
from Geneticore.train_loop import Loop

N_NETS = 1

base_net = Net(net_arr=[1, 64, 64, 1]) #Create a base net to work with
alg = CRGeneticAlg(base_net=base_net, steps=1_000, num_nets=N_NETS) #Create the algorithm and put the base net in it
f_c = FC(num_nets=N_NETS, divisor=10) #Not needed, just makes it easier to track fitness values for later on

alg.make_nets() #The first nets we are going to use

fitnesses = []

t_loop = Loop()