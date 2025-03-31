# Exact Verification of First-Order Methods via Mixed-Integer Linear Programming
This repository is by [Vinit Ranjan](https://vinitranjan1.github.io/), [Jisun Park](https://jisunp515.github.io/), [Stefano Gualandi](https://mate.unipv.it/gualandi/), [Andrea Lodi](https://tech.cornell.edu/people/andrea-lodi/), and [Bartolomeo Stellato](https://stellato.io/) and contains the Python source code to reproduce experiments in our paper [Exact Verification of First-Order Methods via Mixed-Integer Linear Programming](https://arxiv.org/abs/2412.11330).

# Abstract
We present exact mixed-integer linear programming formulations for verifying the performance of first-order methods for parametric quadratic optimization. We formulate the verification problem as a mixed-integer linear program where the objective is to maximize the infinity norm of the fixed-point residual after a given number of iterations. Our approach captures a wide range of gradient, projection, proximal iterations through affine or piecewise affine constraints. We derive tight polyhedral convex hull formulations of the constraints representing the algorithm iterations. To improve the scalability, we develop a custom bound tightening technique combining interval propagation, operator theory, and optimization-based bound tightening. Numerical examples, including linear and quadratic programs from network optimization, sparse coding using Lasso, and optimal control, show that our method provides several orders of magnitude reductions in the worst-case fixed-point residuals, closely matching the true worst-case performance.

# Installation
To install the package, run
```
$ pip install git+https://github.com/stellatogrp/mip_algo_verify
```

## Packages
The main required packages are
```
cvxpy >= 1.2.0
gurobipy >= 12.0.1
PEPit
hydra
```
Free academic licenses for individual use can be obtained from the Gurobi website.

### Running experiments
The main driver for experiments is the [Hydra](https://hydra.cc/docs/intro/) testing framework.
All parameter configurations are found in the `experiments/configs` folder.
To run an experiment, from the `experiments/` folder:
```
python run_experiment.py <example> local
```
where `<example>` is one of the following:
```
ISTA
FISTA
LP
MPC
```

### Results
For each experiment, the results are saved in the corresponding `<example>/outputs/` folder and is timestamped by Hydra with the date and time of the experiment.
The results include the residual values, times, and other auxiliary information along with the experiment log to track outputs from Gurobi and other logged information.
