# -*- coding: utf-8 -*-

from pybrain.rl.environments.cartpole import CartPoleEnvironment, DiscreteBalanceTask, CartPoleRenderer
from pybrain.rl.agents import LearningAgent
from pybrain.rl.experiments import EpisodicExperiment

from matplotlib import pyplot as plt

from scipy import mean

from pybrain.rl.environments import cartpole as cp

from learner import BNL, ActionValueBayesianNetwork

# switch this to True if you want to see the cart balancing the pole (slower)
import numpy

import multiprocessing


#def run(task, parameters):
def run(arg):
    task = arg[0]
    parameters = arg[1]
    #print "run with", task,parameters
    
    
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
    module = ActionValueBayesianNetwork(task.outdim, task.indim)
    

    learner = BNL()
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
        y =experiment.doEpisodes(parameters["EpisodesPerLearn"])
        agent.learn(1)
#        print len(y[0])
        #renderer.drawPlot()
        
        # test performance (these real-world experiences are not used for training)
        if plot:
            env.delay = True
        
        if (episode) % parameters["TestAfter"] == 0:
            #print "Evaluating at episode: ", episode
            
            #experiment.agent = testagent
            r = mean([sum(x) for x in testexperiment.doEpisodes(parameters["TestWith"])])
            
#            xx = testexperiment.doEpisodes(parameters["TestWith"])
#            u = [sum(x) for x in xx]
         
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
            
            
#import sumatra.parameters as p
#import sys
#parameter_file = sys.argv[1]
#parameters = p.SimpleParameterSet(parameter_file)
#
#
#run(["BalanceTask",parameters])