# -*- coding: utf-8 -*-
"""
Created on Wed May  7 18:34:05 2014

@author: nairboon
"""

# -*- coding: utf-8 -*-

from pybrain.rl.environments.cartpole import CartPoleEnvironment, DiscreteBalanceTask, CartPoleRenderer
from pybrain.rl.agents import LearningAgent
from pybrain.rl.experiments import EpisodicExperiment

from matplotlib import pyplot as plt

from scipy import mean


from learner import BNL, ActionValueBayesianNetwork

# switch this to True if you want to see the cart balancing the pole (slower)
render = False

plot = True

plt.ion()


# bayesnetwork learner
env1 = CartPoleEnvironment()

# to inputs state and 4 actions
module1 = ActionValueBayesianNetwork(2, 4)

task1 = DiscreteBalanceTask(env1, 100)
learner1 = BNL()
# % of random actions
learner1.explorer.epsilon = 0.4000001


agent1 = LearningAgent(module1, learner1)
testagent1 = LearningAgent(module1, None)
experiment1 = EpisodicExperiment(task1, agent1)

# NFQ learner
from pybrain.rl.learners.valuebased import NFQ, ActionValueNetwork

env2 = CartPoleEnvironment()

module2 = ActionValueNetwork(2, 4)

task2 = DiscreteBalanceTask(env2, 100)
learner2 = NFQ()
learner2.explorer.epsilon = 0.4

agent2 = LearningAgent(module2, learner2)
testagent2 = LearningAgent(module2, None)
experiment2 = EpisodicExperiment(task2, agent2)




def plotPerformance(values1,values2, fig):
    plt.figure(fig.number)
    plt.clf()
    plt.plot(values1, label="BAYES")
    plt.plot(values2, label='NFQ')
    #print values1,values2

    plt.gcf().canvas.draw()
    plt.legend()
    # Without the next line, the pyplot plot won't actually show up.
    plt.pause(0.001)

performance1 = []
performance2 = []

if plot:
    pf_fig = plt.figure()

while(True):
	# one learning step after one episode of world-interaction
    experiment1.doEpisodes(1)
    agent1.learn(1)
    
    experiment2.doEpisodes(1)
    agent2.learn(1)

    #renderer.drawPlot()
    
    # test performance (these real-world experiences are not used for training)
    if plot:
        env1.delay = True
        
    experiment1.agent = testagent1
    r1 = mean([sum(x) for x in experiment1.doEpisodes(5)])
    env1.delay = False
    testagent1.reset()
    experiment1.agent = agent1
    
    experiment2.agent = testagent2
    r2 = mean([sum(x) for x in experiment2.doEpisodes(5)])
    env2.delay = False
    testagent2.reset()
    experiment2.agent = agent2   
    

    performance1.append(r1)
    performance2.append(r2)

    if plot:
        plotPerformance(performance1, performance2, pf_fig)

    print "reward avg", r1
    print "explorer epsilon", learner1.explorer.epsilon
    print "num episodes", agent1.history.getNumSequences()
    print "update step", len(performance1)
    
    #print "network",   json.dumps(module.bn.net.E, indent=2)