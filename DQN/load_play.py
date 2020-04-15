import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import matplotlib as plt
import sys
import gym

from space_invaders_hybrid import Agent, DeepQNetwork


ALPHA = 0.03
GAMMA = 0.99
EPSILON = 1.0
MAXMEMSIZE = 2000
REPLACE = 10000


if __name__ == "__main__":
    env = gym.make('SpaceInvaders-v0')
    agent = Agent(gamma=GAMMA, epsilon=EPSILON, alpha=ALPHA, maxMemSize=MAXMEMSIZE, replace=REPLACE)

    #load model from "model_parameters.pt" and assigning values to variables
    checkpoint = torch.load("model_parameters.pt")
    agent.Q_eval.load_state_dict(checkpoint["Q_eval"])
    agent.Q_next.load_state_dict(checkpoint["Q_next"])
    agent.Q_eval.optimizer.load_state_dict(checkpoint["optimizer_eval"])
    agent.Q_next.optimizer.load_state_dict(checkpoint["optimizer_next"])
    agent.eps_end = checkpoint(['eps_end'])
    agent.steps = checkpoint(['steps'])
    agent.learn_step_counter = checkpoint(['learn_step_counter'])
    agent.memory = checkpoint(['memory'])
    agent.stacked_frames = checkpoint(['stacked_frames'])
    agent.memCounter = checkpoint(['memCounter'])
    agent.replace_target_counter = checkpoint(['replace_target_counter'])
    games = checkpoint(['game'])

    #play 50 games
    scores = []
    epsHistory = [] 
    numGames = games + 50
    
    for i in range(numGames):
        print('starting game ', i+1, 'epsilon: %.4f' % agent.EPSILON)
        epsHistory.append(agent.EPSILON)
        done = False
        observation = env.reset()
        
        state = agent.stackedState(observation)
        state_skipping = 0
        score = 0
        lastAction = 0 
        
        while not done:
            if state_skipping == 3:                
                action = agent.chooseAction(state)
                state_skipping = 0
            else:
                action = lastAction
                state_skipping += 1
            
            observation_, reward, done, info = env.step(action)
            score += reward
            state_ = agent.stackedState(observation_)
            
            if done and info['ale.lives'] == 0:
                reward = -100

            observation = observation_
            lastAction = action
        
        scores.append(score)
        print('score:',score)