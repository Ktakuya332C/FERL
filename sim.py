from __future__ import division
import numpy as np
from matplotlib import pylab as plt
from tqdm import tqdm, trange
from tools import *

# Paramters
n_hidden = 13
dim_state = 12
dim_action = 40
scale = 0.7
n_key_states = 13
n_train = 30000
n_sample = 100
start_learning_rate = 0.01
last_learning_rate = 0.01
learning_rate = start_learning_rate
start_beta = 1
last_beta = 10
beta = start_beta
ave_per = 1000

# Define MDP
mdp = LargeActionTask(n_key_states, dim_state, dim_action)

# RBM training
rbm = RBM(n_hidden, dim_state, dim_action, scale)
rewards = []
mean_rewards = []
print( "Training start" )
t = trange(1, n_train + 1)
for i in t:
    # Learning rate adaptation
    learning_rate = start_learning_rate * ( last_learning_rate / start_learning_rate ) ** ( i / n_train )
    beta = start_beta * ( last_beta / start_beta ) ** ( i / n_train )
    # Training
    state = mdp.next_key_state()
    action = rbm.play(state, n_sample, beta)
    reward = mdp.reward(action)
    rbm.qlearn(state, action, reward, learning_rate)
    # Save reward
    t.set_description("Reward %d"%reward)
    rewards.append(reward)
    if i % ave_per == 0 and i > 0:
        mean_rewards.append( np.mean(rewards[i-ave_per:i]) )

# Plotting
plt.plot(np.arange(1, 31), mean_rewards)
plt.xlabel("1000s iterations")
plt.ylabel("Average reward")
plt.savefig("result.png")
