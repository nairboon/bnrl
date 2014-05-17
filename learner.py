# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 15:44:50 2014

@author: nairboon
"""


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
#import pandas as pd

class BN:
     def __init__(self):
         self.net = False
         self.burn = True

     def query(self,state,bins):

         # just return a random action 
         if self.burn:
             return random.uniform(-100,100)
             

         #evidence = dict(StateA=state[0], StateB=state[2],StateC=state[1], StateD=state[3])

         evidence = dict(theta=state[0], thetaPrime=state[1],s=state[2], sPrime=state[3])
        # sample the network given evidence
         result = self.net.randomsample(1, evidence)
         
         return result[0]["Action"]
        
         #bins = array([0.0, 1.0, 2.0, 3.0])

         
#         t = pd.cut(ac, 4,labels=False)
#         counts = bincount(t)
#         r = argmax(counts)
         
         a = []
         for x in result:
             a.append(x["Reward"])
             
         i = argmax(a) #position of highest reward
         
         # so the action with highest reward is
         action = result[i]["Action"]
         #print "action:", action
         return action
         # determine which bin it belongs to
         #return digitize([action],bins)
         


class PGMTrainer(Trainer):

    def __init__(self, module, dataset=None):

        Trainer.__init__(self, module)
        #self.setData(dataset)
        self.ds = dataset
        self.learner = PGMLearner()

    def train(self):
        """Train the associated module for one epoch."""
        assert len(self.ds) > 0, "Dataset cannot be empty."

        if len(self.ds) < 10:
#            print "burn"
            self.module.burn = True
            return
        else:
            self.module.burn = False
            
        gbds = []
        for seq in self.ds:
            for state_, action_, reward_ in seq:

                #sample = dict(StateA=state_[0],StateB=state_[2],StateC=state_[1],StateD=state_[3],Action=action_[0],Reward=reward_[0])
                sample = dict(theta=state_[0],thetaPrime=state_[2],s=state_[1],sPrime=state_[3],Action=action_[0],Reward=reward_[0])

                if sample["Reward"] >= -0:
                    gbds.append(sample)
                #print sample["Reward"]

        # sort samples for highest reward
#        bdss = sorted(gbds, key=lambda tup: tup["Reward"],reverse=True)
#        
        #print "BDS: "
        #print json.dumps(gbds, indent=2)
#        print "BDSS: "
#        print json.dumps(bdss, indent=2)
        
        #tokeep = bdss[:max(2,len(bdss)/2)]
        
        #print bds
        # estimate parameters
#        print "data size: ", len(bds),  len(gbds)
        
        
        if len(gbds) < 5: #there was no rewarding action, so nothing to learn
          self.module.burn = True
          return
          
        N = 1000
        if len(gbds) > N:
            #only take the newest N samples

            l = len(gbds)
            gbds = gbds[l-N:]
#            print "new effective set", len(gbds)
        
        skel = GraphSkeleton()
        #load network topology
        #skel.load("net.txt")
        skel.load("workfile")
        skel.toporder()


        # estimate parameters
        self.module.net = self.learner.lg_mle_estimateparams(skel, gbds)

        #self.module.net = learner.lg_estimatebn(bds,pvalparam=0.25)


        # output - toggle comment to see
        #print json.dumps(self.module.net.E, indent=2)
        #print json.dumps(self.module.net.Vdata, indent=2)



class ActionValueBayesianNetwork(Module, ActionValueInterface):
    def __init__(self, dimState, numActions, name=None):
        Module.__init__(self, dimState, 1, name)
        self.network = buildNetwork(dimState + numActions, dimState + numActions, 1)
        self.numActions = numActions
        self.numStates = dimState
        self.bn = BN()
        self.bn.numActions = numActions

    def _forwardImplementation(self, inbuf, outbuf):
        """ takes a vector of length 1 (the state coordinate) and return
            the action with the maximum value over all actions for this state.
        """
        outbuf[0] = self.getMaxAction(asarray(inbuf))

    def getMaxAction(self, state):
        """ Return the action with the maximal value for the given state. """
        #print argmax(self.getActionValues(state))
       # return argmax(self.getActionValues(state))
        r = self.bn.query(state,self.numActions)
        #print r
        return r

    def getActionValues(self, state):
        """ Run forward activation for each of the actions and returns all values. """
        #values = array([self.bn.query(state, i) for i in range(self.numActions)])
        #print values

        #return values



class BNL(ValueBasedLearner):
    """ Bayesian Network learning"""

    def __init__(self):
        ValueBasedLearner.__init__(self)
        self.gamma = 0.9

    def learn(self):
        # convert reinforcement dataset to NFQ supervised dataset
#        supervised = SupervisedDataSet(self.module.network.indim, 1)
#        uv = UnsupervisedDataSet(self.module.network.indim)
#        for seq in self.dataset:
#            lastexperience = None
#            for state, action, reward in seq:
#                if not lastexperience:
#                    # delay each experience in sequence by one
#                    lastexperience = (state, action, reward)
#                    continue
#
#                # use experience from last timestep to do Q update
#                (state_, action_, reward_) = lastexperience
#                #inp = r_[state_, one_to_n(action_[0], self.module.numActions)]
#                #inp = lastexperience
#                #tgt = reward_ + self.gamma * self.module.getMaxAction(state)
#                #supervised.addSample(inp, tgt)
#
#                #uv.addSample(inp)
#                # update last experience with current one
#                lastexperience = (state, action, reward)

        # train module with backprop/rprop on dataset
        #trainer = RPropMinusTrainer(self.module.network, dataset=supervised, batchlearning=True, verbose=False)


        trainer = PGMTrainer(self.module.bn, dataset=self.dataset)
        # alternative: backprop, was not as stable as rprop
        # trainer = BackpropTrainer(self.module.network, dataset=supervised, learningrate=0.01, batchlearning=True, verbose=True)

        trainer.trainEpochs(1)