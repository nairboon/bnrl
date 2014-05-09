# -*- coding: utf-8 -*-

from pybrain.rl.environments.cartpole import CartPoleEnvironment, DiscreteBalanceTask, CartPoleRenderer
from pybrain.rl.agents import LearningAgent
from pybrain.rl.experiments import EpisodicExperiment

from matplotlib import pyplot as plt

from scipy import mean


from learner import BNL, ActionValueBayesianNetwork

# switch this to True if you want to see the cart balancing the pole (slower)

def run(parameters):
    print "run with", parameters
    
    
    render = False    
    plot = False
    
    plt.ion()
    
    env = CartPoleEnvironment()
    if render:
        renderer = CartPoleRenderer()
        env.setRenderer(renderer)
        renderer.start()
    
    # to inputs state and 4 actions
    module = ActionValueBayesianNetwork(2, 3)
    
    task = DiscreteBalanceTask(env, parameters["MaxRunsPerEpisode"])
    learner = BNL()
    # % of random actions
    learner.explorer.epsilon = parameters["ExplorerEpsilon"]
    
    
    agent = LearningAgent(module, learner)
    testagent = LearningAgent(module, None)
    experiment = EpisodicExperiment(task, agent)
    
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
    
    for episode in range(0,parameters["MaxTotalEpisodes"]):
    	# one learning step after one episode of world-interaction
        experiment.doEpisodes(parameters["EpisodesPerLearn"])
        agent.learn(1)
    
        #renderer.drawPlot()
        
        # test performance (these real-world experiences are not used for training)
        if plot:
            env.delay = True
        
        if (episode) % parameters["TestAfter"] == 0:
            print "Evaluating at episode: ", episode
            
            experiment.agent = testagent
            r = mean([sum(x) for x in experiment.doEpisodes(parameters["TestWith"])])
            
            env.delay = False
            testagent.reset()
            experiment.agent = agent
        
            performance.append(r)
            if plot:
                plotPerformance(performance, pf_fig)
        
            print "reward avg", r
            print "explorer epsilon", learner.explorer.epsilon
            print "num episodes", agent.history.getNumSequences()
            print "update step", len(performance)
            
    print "done"
    return performance
            
        #print "network",   json.dumps(module.bn.net.E, indent=2)