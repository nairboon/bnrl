# -*- coding: utf-8 -*-
"""
Created on Fri May 16 14:05:09 2014

@author: nairboon
"""

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


from pybrain.rl.learners.valuebased.valuebased import ValueBasedLearner
from pybrain.datasets import SupervisedDataSet, UnsupervisedDataSet

from scipy import argmax, array, r_, asarray
from pybrain.utilities import abstractMethod
from pybrain.structure.modules import Table, Module
from pybrain.structure.parametercontainer import ParameterContainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.utilities import one_to_n

from pybrain.rl.learners.valuebased.interface import ActionValueInterface


from pybrain.supervised.trainers import Trainer


from libpgm.pgmlearner import PGMLearner
from libpgm.graphskeleton import GraphSkeleton

import json

from numpy import digitize, bincount
from scipy import random


class RAND(ValueBasedLearner):
    """ Bayesian Network learning"""

    def __init__(self):
        ValueBasedLearner.__init__(self)
        self.gamma = 0.9

#    def learnX(self, ds):
#        self.dataset = ds

    def add_ds(self,ds):
        print "adding ", len(ds)
        for seq in ds:
            for state, action, reward in seq:
                self.dataset.addSample(state,action,reward)
        
    def learn(self):
        print "ds: ", len(self.dataset)
        #print self.dataset
        
        data = []
        
        rw = []
        
        
        bestreward = -100
        for seq in self.dataset:
            for state_, action_, reward_ in seq:
                if reward_[0] > bestreward:
                    bestreward = reward_[0]
                
                # find limit for theta
                 
        print "bestrw", bestreward
        nds = []
        lt=[]
        ls = []
        ltv =[]
        lsv=[]
        
        i = 0
        for seq in self.dataset:
            for state_, action_, reward_ in seq:
#                if reward_[0] == 0:
#                    print state_, action_, reward_
                #print state_, reward_
                if reward_[0] == bestreward:
                    ns = (state_, action_[0], reward_[0])
                    nds.append(ns)
#                    print state_[0], state_[2], reward_[0], bestreward
                    
                    t = state_[0]
                    tv= state_[1]
                    
                    s = state_[2] 
                    sv = state_[3]    
                    if t > 0.05:
                        print "hmmm,", i, t
                        #raise Exception(i)
                        
                    i += 1
                    lt.append(t)
                    ls.append(s)
                    ltv.append(tv)
                    lsv.append(sv)
        
        
        limits = dict(theta=[min(lt),max(lt)],s=[min(ls),max(ls)],thetaV=[min(ltv),max(ltv)],sV=[min(lsv),max(lsv)])
        
        print "limits: ", limits
                    
#        print "all good things:", nds
                    
                
                
        #convert ds
        for seq in self.dataset:
            for state_, action_, reward_ in seq:
                
#                sample = dict(theta=state_[0],thetaPrime=state_[1],s=state_[2],sPrime=state_[3],Action=action_[0],Reward=reward_[0])
#
#                
#                dtpo = min( abs(sample["thetaPrime"] - limits["theta"][0]), abs(sample["thetaPrime"] - limits["theta"][1]))
#                dto = min( abs(sample["theta"] - limits["theta"][0]), abs(sample["theta"] - limits["theta"][1]))
#                dspo = min( abs(sample["sPrime"] - limits["s"][0]), abs(sample["sPrime"] - limits["s"][1]))
#                dso = min( abs(sample["s"] - limits["s"][0]), abs(sample["s"] - limits["s"][1]))
#                             
#               #print dspo, dso
#                
#                netsample = dict(theta=sample["theta"],s=sample["s"],Action=sample["Action"],Reward=sample["Reward"])
#                # did this action improve theta or s??
#                if dtpo <= dto or dspo <= dso: #yes it did            
##                    data.append(netsample)
#                    rw.append(sample["Reward"])
                sample = dict(theta=state_[0],thetaV=state_[1],s=state_[2],sV=state_[3],Action=action_[0],Reward=reward_[0])

                #print state_, action_, reward_
                #print sample
                if sample["Reward"] != 990:
                    data.append(sample)
                    if numpy.random.random() >= 9.1:
                        continue
                
                
          

        import matplotlib.pyplot as plt
        import pandas as pd
        df = pd.DataFrame(rw)
#        print df        
        
#        plt.figure()
#        df[0].diff().hist()
        
        # instantiate my learner 
        learner = PGMLearner()
        
        # estimate parameters
        rbn = []
        for i in range(0,1):
            result = learner.lg_constraint_estimatestruct(data,bins=10, pvalparam=0.05)
            rbn.append(result)
            print len(result.E), result.E
            
        result = rbn[0]
        
        # output - toggle comment to see
       

        print json.dumps(result.V, indent=2)
        print len(result.E), "Edges", result.E
        
        import pydot

        # this time, in graph_type we specify we want a DIrected GRAPH
        graph = pydot.Dot(graph_type='digraph')
        nd = {}
        for n in result.V:
            nd[n] = pydot.Node(n)
            graph.add_node(nd[n])
            
        for e in result.E:
            
            graph.add_edge(pydot.Edge(nd[e[0]], nd[e[1]]))
            
        graph.write_png('eg.png')
        from IPython.display import Image
        Image('eg.png')
        
        
        f = open('workfile', 'w')
        f.write("{\n \"V\":")
        f.write(json.dumps(result.V))
        f.write(",\n \"E\":")
        f.write(json.dumps(result.E))
        f.write("}")
        f.close()
        
        skel = GraphSkeleton()
        skel.load("workfile")
        
        # topologically order graphskeleton
        skel.toporder()
        

        return

class ActionValueRAND(Module, ActionValueInterface):
    def __init__(self, dimState, numActions, name=None):
        Module.__init__(self, dimState, 1, name)
        self.network = buildNetwork(dimState + numActions, dimState + numActions, 1)
        self.numActions = numActions
        self.numStates = dimState

    def _forwardImplementation(self, inbuf, outbuf):
        """ takes a vector of length 1 (the state coordinate) and return
            the action with the maximum value over all actions for this state.
        """
        outbuf[0] = self.getMaxAction(asarray(inbuf))

    def getMaxAction(self, state):
        """ Return the action with the maximal value for the given state. """
        #print argmax(self.getActionValues(state))
       # return argmax(self.getActionValues(state))
        
       
        return random.uniform(-50,50)

    def getActionValues(self, state):
        """ Run forward activation for each of the actions and returns all values. """
        #values = array([self.bn.query(state, i) for i in range(self.numActions)])
        #print values

        #return values

#def run(task, parameters):
def run(arg):
    task = arg[0]
    parameters = arg[1]
    #print "run with", task,parameters
    
    
    seed = parameters["seed"]
   

    process_id = hash(multiprocessing.current_process()._identity)
    numpy.random.seed(seed)
    
    render = False    
    plot = False
    
    plt.ion()
    
    env = CartPoleEnvironment()
    env.randomInitialization = False
    if render:
        renderer = CartPoleRenderer()
        env.setRenderer(renderer)
        renderer.start()
    
    task_class = getattr(cp, task)
    task = task_class(env, 50)

    #print "dim: ", task.indim, task.outdim
    
    # to inputs state and 4 actions
    bmodule = ActionValueRAND(task.outdim, task.indim)
    rlearner = RAND()

    blearner = RAND()
    # % of random actions
    
    bagent = LearningAgent(bmodule, rlearner)
    
    from pybrain.tools.shortcuts import buildNetwork
    from pybrain.rl.agents import OptimizationAgent
    from pybrain.optimization import PGPE

    module = buildNetwork(task.outdim, task.indim, bias=False)
    # create agent with controller and learner (and its options)

    # % of random actions
    #learner.explorer.epsilon = parameters["ExplorerEpsilon"]
    
    
    agent = OptimizationAgent(module, PGPE(storeAllEvaluations = True,storeAllEvaluated=True, maxEvaluations=None, verbose=False))


    
    
    testagent = LearningAgent(module, None)
    pgpeexperiment = EpisodicExperiment(task, agent)
    randexperiment = EpisodicExperiment(task, bagent)


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
    
    ## train pgpe
    for episode in range(0,50):
    	# one learning step after one episode of world-interaction
        y =pgpeexperiment.doEpisodes(1)
        
    be, bf = agent.learner._bestFound()
    print be,bf
    
    print "generate data"
    be.numActions = 1
    gdagent = LearningAgent(be, blearner)
    experiment = EpisodicExperiment(task, gdagent)
    
    for episode in range(0,1000):
#        print episode, " of 1000"
    	# one learning step after one episode of world-interaction
        y =experiment.doEpisodes(1)
        
#        print y
        x = randexperiment.doEpisodes(1)
#        print len(y[0])
        #renderer.drawPlot()
        
        # test performance (these real-world experiences are not used for training)
        if plot:
            env.delay = True
        

        l = 5
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
        #print resList
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
            
    blearner.add_ds(rlearner.dataset)
    
    blearner.learn()
    #blearner.learnX(agent.learner._allEvaluated)
    print "done"
    return performance
            
        #print "network",   json.dumps(module.bn.net.E, indent=2)
            
            
import sumatra.parameters as p
import sys
parameter_file = sys.argv[1]
parameters = p.SimpleParameterSet(parameter_file)


run(["BalanceTask",parameters])