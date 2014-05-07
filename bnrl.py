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

env = CartPoleEnvironment()
if render:
    renderer = CartPoleRenderer()
    env.setRenderer(renderer)
    renderer.start()

# to inputs state and 4 actions
module = ActionValueBayesianNetwork(2, 4)

task = DiscreteBalanceTask(env, 100)
learner = BNL()
# % of random actions
learner.explorer.epsilon = 0.2000001


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

while(True):
	# one learning step after one episode of world-interaction
    experiment.doEpisodes(1)
    agent.learn(1)

    #renderer.drawPlot()
    
    # test performance (these real-world experiences are not used for training)
    if plot:
        env.delay = True
    experiment.agent = testagent
    r = mean([sum(x) for x in experiment.doEpisodes(5)])
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
    
    #print "network",   json.dumps(module.bn.net.E, indent=2)