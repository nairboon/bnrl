# -*- coding: utf-8 -*-
"""
Created on Mon May 12 14:52:55 2014

@author: nairboon
"""

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
from pybrain.rl.learners.directsearch import ENAC


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
    
    testtask = task_class(env, parameters["MaxRunsPerEpisodeTest"],desiredValue=None)

    #print "dim: ", task.indim, task.outdim

    from pybrain.tools.shortcuts import buildNetwork
    from pybrain.rl.agents import OptimizationAgent
    from pybrain.optimization import PGPE

    module = buildNetwork(task.outdim, task.indim, bias=False)
    # create agent with controller and learner (and its options)

    # % of random actions
    #learner.explorer.epsilon = parameters["ExplorerEpsilon"]
    
    
    agent = OptimizationAgent(module, PGPE(storeAllEvaluations = True,storeAllEvaluated=False, maxEvaluations=None,desiredEvaluation=1, verbose=False))
#
#    print agent
#    from pprint import pprint
#    pprint (vars(agent.learner))
    
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
        #agent.learn(1)
    
        #renderer.drawPlot()
        
        # test performance (these real-world experiences are not used for training)
        if plot:
            env.delay = True
        
        if (episode) % parameters["TestAfter"] == 0:
            #print "Evaluating at episode: ", episode
            
            #experiment.agent = testagent
            #r = mean([sum(x) for x in testexperiment.doEpisodes(parameters["TestWith"])])
            #for i in range(0,parameters["TestWith"]):
#            y = testexperiment.doEpisodes(1)
#            print (agent.learner._allEvaluated)
#                
#            
#            from pprint import pprint
#            pprint (vars(task))
                
            l = parameters["TestWith"]
            
            task.N = parameters["MaxRunsPerEpisodeTest"]
            experiment.doEpisodes(l)
            task.N = parameters["MaxRunsPerEpisode"]

            resList = (agent.learner._allEvaluations)[-l:-1]
            
#            print agent.learner._allEvaluations
            from scipy import array

            rLen = len(resList)
            avReward = array(resList).sum()/rLen
#            print avReward
#            print resList
#            exit(0)
#            print("Parameters:", agent.learner._bestFound())
#            print(
#                " Evaluation:", episode,
#                " BestReward:", agent.learner.bestEvaluation,
#                " AverageReward:", avReward)
#            if agent.learner.bestEvaluation == 0:
#                
#                print resList[-20:-1]
#                print "done"
#                break
            performance.append(avReward)
            

            env.delay = False
            testagent.reset()
            #experiment.agent = agent
        
#            performance.append(r)
            if plot:
                plotPerformance(performance, pf_fig)
        
#            print "reward avg", r
#            print "explorer epsilon", learner.explorer.epsilon
#            print "num episodes", agent.history.getNumSequences()
#            print "update step", len(performance)
            
#    print "done"
    return performance
            
        #print "network",   json.dumps(module.bn.net.E, indent=2)
            
            
#import sumatra.parameters as p
#import sys
#parameter_file = sys.argv[1]
#parameters = p.SimpleParameterSet(parameter_file)
#
#
#run(["BalanceTask",parameters])