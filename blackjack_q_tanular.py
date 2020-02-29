import gym
import numpy as np
from collections import defaultdict
import random
import sys

#Blackjack using tabular Q-learning, Q table will be a defaultdictionary
#if the state is not yet in the dictionary, it will be added

env = gym.make('Blackjack-v0')
 


#hyperparameters
EPSILON = 0.9
EPSILON_MIN = 0.01
TOTAL_EP = 500000
MAX_STEP = 100
LR_RATE = 0.009 #really small learning rate, but this finds the optimal policy
GAMMA = 0.96


def epsilon_greedy(Q,state,eps):
    """Epsilon greedy policy - with probability >0.9 it will do a random action, otherwise 
       chooses the action corresponding the highest state-action value. Epsilon decays as we progress further.
        # input: the q table and a correspond state of the simulation
        # output: action selected by the policy"""
    if  np.random.uniform(0, 1) > eps:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state])
    return action

def q_learning(Q, state, action, state2, reward):
    """Updates the Q-table with the latest experience.
        # input: Q table, state and action pair, next state and rewards (from the environment)
        # no output, the Q table will be updated"""
    current = Q[state][action] 
    target = reward + GAMMA*np.max(Q[state2]) if state2 is not None else 0
    Q[state][action] = current + LR_RATE*(target-current)


def simulation():
    """Runs the simulation using the hyperparameters"""
    actionspace = env.action_space.n
    Q = defaultdict(lambda: np.zeros(actionspace))
    rewards = 0
    for episode in range(1,TOTAL_EP+1):
        #if episode % 100 == 0:
        #    print("\rEpisode {}/{}".format(episode, TOTAL_EP), end="")
        #    sys.stdout.flush()

        t = 0
        state = env.reset()
        eps = max(1.0 / episode ,EPSILON_MIN)  

        while t < MAX_STEP:
            action = epsilon_greedy(Q, state, eps)
            state2, reward, done, info = env.step(action)

            q_learning(Q, state, action, state2, reward)
            state = state2
            t += 1
            rewards +=1

            if done:
                break
    return Q            

Q = simulation()
print("")
for key,value in sorted(Q.items()):
    print(key, np.argmax(value))
    
  
