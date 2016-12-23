# FERL

## Overview
This is a implementation of Free Energy based Reinforcement Learning (FERL). Especially implements the Large Action Task described in 
* Brian Sallans, Geoffrey E. Hinton, Reinforcement Learning with Factored States and Actions

## Requirement
* python2.7
* starndard scientific computing libraries like numpy, scipy, matplotlib
* tqdm

## Execution
`python sim.py`
with execute the script and save "result.png" in the same folder

## Notice
Learning rate is different from the literature because the q value diverge when I set the learning rate value same to the literature.
