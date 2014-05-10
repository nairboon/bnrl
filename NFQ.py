# -*- coding: utf-8 -*-
"""
Created on Fri May  9 14:36:24 2014

@author: nairboon
"""

# -*- coding: utf-8 -*-

from pybrain.rl.environments.cartpole import CartPoleEnvironment, DiscreteBalanceTask, CartPoleRenderer
from pybrain.rl.agents import LearningAgent
from pybrain.rl.experiments import EpisodicExperiment

from matplotlib import pyplot as plt

from scipy import mean

from pybrain.rl.learners.valuebased import NFQ, ActionValueNetwork, Q

from pybrain.rl.environments import cartpole as cp

from learner import BNL, ActionValueBayesianNetwork

# switch this to True if you want to see the cart balancing the pole (slower)

import numpy

import multiprocessing

#def run(task, parameters):
def run(arg):
    task = arg[0]
    parameters = arg[1]
    #print "run with", parameters
    
    seed = parameters["seed"]
   

    process_id = hash(multiprocessing.current_process()._identity)
    numpy.random.seed(seed + process_id)


    
    
    render = False    
    plot = False
    
    plt.ion()
    
    env = CartPoleEnvironment()
    if render:
        renderer = CartPoleRenderer()
        env.setRenderer(renderer)
        renderer.start()
    
    task_class = getattr(cp, task)
    task = task_class(env, parameters["MaxRunsPerEpisode"])
    testtask = task_class(env, parameters["MaxRunsPerEpisodeTest"])

    #print "dim: ", task.indim, task.outdim
    
    # to inputs state and 4 actions
    module = ActionValueNetwork(task.outdim, task.indim)
    

    learner = NFQ()
    # % of random actions
    learner.explorer.epsilon = parameters["ExplorerEpsilon"]
    
    
    agent = LearningAgent(module, learner)
    testagent = LearningAgent(module, None)
    experiment = EpisodicExperiment(task, agent)
    testexperiment = EpisodicExperiment(testtask, testagent)

    
    def plotPerformance(values, fig):
        plt.figure(fig.number)
        plt.clf()
        plt.plot(values, 'o-')
        plt.gcf().canvas.draw()
        # Without the next line, the pyplot plot won't actually show up.
        plt.pause(0.001)
    
    performance = []
    
    if plot:
        pf_fig = plt.figure()
    
    m = parameters["MaxTotalEpisodes"]/parameters["EpisodesPerLearn"]
    for episode in range(0,m):
    	# one learning step after one episode of world-interaction
        experiment.doEpisodes(parameters["EpisodesPerLearn"])
        agent.learn(1)
    
        #renderer.drawPlot()
        
        # test performance (these real-world experiences are not used for training)
        if plot:
            env.delay = True
        
        if (episode) % parameters["TestAfter"] == 0:
            #print "Evaluating at episode: ", episode
            
            #experiment.agent = testagent
            r = mean([sum(x) for x in testexperiment.doEpisodes(parameters["TestWith"])])
            
            env.delay = False
            testagent.reset()
            #experiment.agent = agent
        
            performance.append(r)
            if plot:
                plotPerformance(performance, pf_fig)
        
#            print "reward avg", r
#            print "explorer epsilon", learner.explorer.epsilon
#            print "num episodes", agent.history.getNumSequences()
#            print "update step", len(performance)
            
#    print "done"
    return performance
            
        #print "network",   json.dumps(module.bn.net.E, indent=2)