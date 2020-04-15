#pip3 install cmake gym gym[atari] numpy torch matplotlob

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import matplotlib as plt
import sys

import gym

class DeepQNetwork(nn.Module):
    def __init__(self,ALPHA):
        super(DeepQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 8, stride=4, padding=1) #3 is for rgb, 32 filter, 8*8 kernel
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2) 
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(128*19*8, 512) 
        self.fc2 = nn.Linear(512, 6)
        self.optimizer = optim.RMSprop(self.parameters(), lr=ALPHA) 
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state = T.Tensor(state).to(self.device)
        state = state.view(-1, 3, 185, 95) 
        state = F.relu(self.conv1(state))
        state = F.relu(self.conv2(state))
        state = F.relu(self.conv3(state))
        state = state.view(-1, 128*19*8) 
        state = F.relu(self.fc1(state))
        actions = self.fc2(state)
        return actions


class Agent(object):
    def __init__(self, gamma, epsilon, alpha, maxMemSize, epsEnd=0.05,
                 replace=10000,actionSpace=[0,1,2,3,4,5]):
        self.GAMMA = gamma
        self.EPSILON = epsilon
        self.EPS_END = epsEnd
        self.ALPHA = alpha
        self.actionSpace = actionSpace
        self.memSize = maxMemSize
        self.steps = 0
        self.learn_step_counter = 0 
        self.memory = []
        self.stacked_frames = []
        self.memCounter = 0
        self.replace_target_counter = replace
        self.Q_eval = DeepQNetwork(alpha) 
        self.Q_next = DeepQNetwork(alpha) 

    def getScreen(self, state):
        return np.mean(state[15:200,30:125], -1) / 255

    def stackedState(self, state):
        state = self.getScreen(state)
        if len(self.stacked_frames) == 0:
            self.stacked_frames.append(state)
            self.stacked_frames.append(state)
            self.stacked_frames.append(state)
        else:
            self.stacked_frames.pop(0)
            self.stacked_frames.append(state)
        
        state = np.stack(self.stacked_frames, axis=2)
        return state 


    def storeTransition(self, state, action, reward, next_state):
        if self.memCounter < self.memSize:
            self.memory.append([state,action,reward,next_state])
        else:
            self.memory[self.memCounter%self.memSize] = [state,action,reward,next_state]
        self.memCounter += 1    

    def chooseAction(self, observation):
        rand = np.random.random()
        actions = self.Q_eval.forward(observation) 
        
       
        if rand < 1-self.EPSILON:
            action = T.argmax(actions[0]).item()
            
            #TODO: actions[1] represents the 2rd row of the matrix, we will return argmax of this row
        else:
            action = np.random.choice(self.actionSpace)
        self.steps += 1
        return action

    def learn(self, batch_size):
        self.Q_eval.optimizer.zero_grad()
        if self.replace_target_counter is not None and self.learn_step_counter % self.replace_target_counter == 0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict())
            print("\033[91m"+"Weights have been copied!"+"\033[0m")
        
        if self.memCounter+batch_size < self.memSize:
            memStart = int(np.random.choice(range(self.memCounter)))
        else:
            memStart = int(np.random.choice(range(self.memSize-batch_size-1)))
        miniBatch=self.memory[memStart:memStart+batch_size]
        memory = np.array(miniBatch)

        Qpred = self.Q_eval.forward(list(memory[:,0][:])).to(self.Q_eval.device)
        Qnext = self.Q_next.forward(list(memory[:,3][:])).to(self.Q_eval.device)

        maxA = T.argmax(Qnext, dim=1).to(self.Q_eval.device)
        rewards = T.Tensor(list(memory[:, 2])).to(self.Q_eval.device)
        Qtarget = Qpred.clone()
        indices = np.arange(batch_size) #[0,1,2, ..., 31]
        Qtarget[indices,maxA] = rewards + self.GAMMA*T.max(Qnext[1])

        if self.steps > 500:
            #this could be done in chooseAction 
            if self.EPSILON - 1e-4 > self.EPS_END:
                self.EPSILON -= 1e-4
            else:
                self.EPSILON = self.EPS_END

        loss = self.Q_eval.loss(Qtarget, Qpred).to(self.Q_eval.device)
        loss.backward() 
        self.Q_eval.optimizer.step() 
        self.learn_step_counter +=1
        #print("\rlearning called: {}".format(self.learn_step_counter), end="")
        #sys.stdout.flush()

if __name__ == '__main__':
    env = gym.make('SpaceInvaders-v0')
    agent = Agent(gamma=0.99, epsilon=1.0, alpha=0.03, maxMemSize=2000, replace=10000)

    def plotLearning(x, scores, epsilons, filename, lines=None):
        fig=plt.figure()
        ax=fig.add_subplot(111, label="1")
        ax2=fig.add_subplot(111, label="2", frame_on=False)

        ax.plot(x, epsilons, color="C0")
        ax.set_xlabel("Game", color="C0")
        ax.set_ylabel("Epsilon", color="C0")
        ax.tick_params(axis='x', colors="C0")
        ax.tick_params(axis='y', colors="C0")

        N = len(scores)
        running_avg = np.empty(N)
        for t in range(N):
            running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

        ax2.scatter(x, running_avg, color="C1")
        #ax2.xaxis.tick_top()
        ax2.axes.get_xaxis().set_visible(False)
        ax2.yaxis.tick_right()
        #ax2.set_xlabel('x label 2', color="C1")
        ax2.set_ylabel('Score', color="C1")
        #ax2.xaxis.set_label_position('top')
        ax2.yaxis.set_label_position('right')
        #ax2.tick_params(axis='x', colors="C1")
        ax2.tick_params(axis='y', colors="C1")

        if lines is not None:
            for line in lines:
                plt.axvline(x=line)

        plt.savefig(filename) 

    while agent.memCounter < agent.memSize:
        observation = env.reset()
        done = False
        state = agent.stackedState(observation)
        while not done:
            # 0 no action, 1 fire, 2 move right, 3 move left, 4 move right fire, 5move left fire
            action = env.action_space.sample() 
            observation_, reward, done, info = env.step(action)
            state_ = agent.stackedState(observation_)           
            if done and info['ale.lives'] == 0:
                reward = -100

            agent.storeTransition(state, action, reward, state_)
            observation = observation_
    print("\033[91m" + 'done initialising memory' + "\033[0m")

    scores = []
    epsHistory = [] 
    numGames = 50
    batch_size = 32 

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
            agent.storeTransition(state, action, reward, state_)
            observation = observation_
            agent.learn(batch_size)
            lastAction = action
        
        scores.append(score)
        print('score:',score)
        
        
    T.save({
        'game': i,
        'Q_eval': agent.Q_eval.state_dict(),
        'Q_next': agent.Q_next.state_dict(),
        'optimizer_eval': agent.Q_eval.optimizer.state_dict(),
        'optimizer_next': agent.Q_next.optimizer.state_dict(),
        'eps_end': agent.eps_end,
        'steps' : agent.steps,
        'learn_step_counter' : agent.learn_step_counter,
        'memory': agent.memory,
        'stacked_frames': agent.stacked_frames,
        'memCounter': agent.memCounter,
        'replace_target_counter': agent.replace_target_counter,
    }, "model_parameters.pt")
    
    x = [i+1 for i in range(numGames)]
    fileName = str(numGames) + 'Games' + 'Gamma' + str(agent.GAMMA) + 'Alpha' + str(agent.ALPHA) + 'Memory' + str(agent.memSize)+ '.png'
    plotLearning(x, scores, epsHistory, fileName)
    
    
