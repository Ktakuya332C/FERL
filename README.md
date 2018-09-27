FERL
----
An implementation of the simplest kind of Free Energy based Reinforcement Learning (FERL) algorithm. Especially, this repository implements the Large Action Task described in the following literature.

* Brian Sallans, Geoffrey E. Hinton (2004), Reinforcement Learning with Factored States and Actions, J. Mach. Learn. Res., 1063-1088

## Requirements
* preferably, python 2.7
* starndard scientific computing libraries like numpy, scipy, matplotlib
* [tqdm](https://pypi.org/project/tqdm/)

## Execution
```
cd FERL
python sim.py
```
will execute a script, and save `result.png` in the current folder. The picture shows average rewards per 1000 epochs acquired by an agent in the simulation.

## Notes
* The implementation of the Large Action Task described in the original literature differs from this implementation. In the original literature, the current state is randomly generated, but in this implementation, the state is randomly chosen only from the predefined key states. I tried to reproduce the original implementation many times, but I could not make it work.
* Learning rate used in this implementation is different from the literature because Q value diverges when I set the same learning rate to the literature.
