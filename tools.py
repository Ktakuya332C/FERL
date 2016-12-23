from __future__ import division
import time
from math import exp, log, floor
from operator import add, mul
from random import random, randint
import numpy as np
from matplotlib import pylab as plt

def sig(x):
    if x < -700:
        return 0.0
    else:
        return 1 / ( 1 + np.exp(-x) )
sig_vec = np.vectorize(sig)

def samp(p):
    if random() < p:
        return 1
    else:
        return 0
samp_vec = np.vectorize(samp)

def logexp(x):
    if x > 700:
        return x
    else:
        return log(1+exp(x))
logexp_vec = np.vectorize(logexp)

def safe_log(x):
    if x < 1e-32:
        x = 1e-32
    return log(x)
log_vec = np.vectorize(safe_log)


def similarity(m, y):
    """
    Input
    -------
    m: binary matrix, shape = (data_len, n_feature)
    y: binary matrix, shape = (n_feature)
    
    Output
    ---------
    similarity between each row and y, the number of same entries over each row and y
    """
    return np.sum((m + y + 1) %2, axis=1)

class LargeActionTask:
    
    def __init__(self, n_key_states, dim_state, dim_action):
        self.n_key_states = n_key_states
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.key_states = np.random.randint(0, 2, size=(n_key_states, dim_state))
        self.key_actions = np.random.randint(0, 2, size=(n_key_states, dim_action))
        self.current_state = np.random.randint(0, 2, self.dim_state)
    
    def next_state(self):
        self.current_state = np.random.randint(0, 2, self.dim_state)
        return self.current_state
    
    def next_key_state(self):
        ind = np.random.randint(0, self.n_key_states)
        return self.key_states[ind, :]
    
    def reward(self, action):
        ind = np.argmax(similarity(self.key_states, self.current_state))
        return np.sum(action == self.key_actions[ind, :])
    
    def optimal_action(self):
        ind = np.argmax(similarity(self.key_states, self.current_state))
        return self.key_actions[ind, :]

class RBM:
    
    def __init__(self, n_hidden, dim_state, dim_action, scale=None):
        self.n_hidden = n_hidden
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.n_visible = dim_state + dim_action
        self.scale = scale
        self.w = np.random.uniform(low=-self.scale, high=self.scale, size=(n_hidden, dim_state))
        self.u = np.random.uniform(low=-self.scale, high=self.scale, size=(n_hidden, dim_action))

    def tau(self, s, a):
        return np.dot(self.w, s) + np.dot(self.u, a)

    def lam(self, s, a):
        return -logexp_vec(self.tau(s, a))
    
    def q(self, s, a):
        return -np.sum(self.lam(s, a))
    
    def play(self, s, n_sample, beta):
        # First deterministic initialization
        h = samp_vec(sig_vec(beta * np.dot(self.w, s)))
        a = samp_vec(sig_vec(beta * np.dot(self.u.T, h)))
        
        # Gibbs sampling
        for i in range(n_sample):
            h = samp_vec(sig_vec(beta * self.tau(s, a)))
            a = samp_vec(sig_vec(beta * np.dot(self.u.T, h)))

        return a
    
    def qlearn(self, s, a, r, lr):
        # q learning with gamma = 0
        ph = sig_vec(self.tau(s, a))
        self.w += lr * (r - self.q(s, a)) * np.outer(ph, s)
        self.u += lr * (r - self.q(s, a)) * np.outer(ph, a)