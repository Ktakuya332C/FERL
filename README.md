# FERL

## Overview
This is a implementation of Free Energy based Reinforcement Learning (FERL). Especially implements the Large Action Task (Not exctly same) described in 
* Brian Sallans, Geoffrey E. Hinton (2004), Reinforcement Learning with Factored States and Actions, J. Mach. Learn. Res., 1063-1088

## Requirement
* python2.7
* starndard scientific computing libraries like numpy, scipy, matplotlib
* tqdm

## Execution
`python sim.py`
will execute the script and save "result.png" in the same folder

## Notice
The crucial difference from the original literature is that the Large Action Task is implemented differently from the original literature. In the original literature, current state is randomly generated but in this implementation, current state is randomly chosen only from the key states. I tried the original implementation many times but I could not make it work. 
Learning rate is different from the literature because the q value diverges when I set the learning rate value same to the literature.
