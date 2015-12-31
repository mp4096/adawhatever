# `adawhatever`
A collection of various stochastic gradient descent (SGD) solvers implemented in MATLAB:
* Vanilla SGD
* AdaGrad (vanilla / with decay)
* Adadelta
* Adam
* Adamax

## Introduction
Stochastic gradient descent is a state of the art optimisation method in machine learning. It suits the concept of learning on many data points very well and outperforms many _theoretically_ superior second-order methods.

All fancy SGD solvers are readily available in every machine learning framework, e.g. [Lasagne](https://github.com/Lasagne/Lasagne). However, machine learning usage is not the goal of this repo. Instead, it attempts to facilitate transfer of novel algorithms from machine learning to other, mostly engineering, applications. These communities tend to speak MATLAB and not Python; this motivates the language choice.

Furthermore, 'classic' optimisation terms are used instead of machine learning lingo, e.g.:
* learning rate -> step size
* parameters -> decision variables
* hyperparameters ~ solver parameters

## How to use it
All solvers require the stochastic gradient of the objective `sg`, initial value of the decision variables `x0`, number of iterations `nIter` and the indices of the stochastic gradient that should be used at each iteration `idxSG`. Furthermore, each solver requires its specific solver parameters. See MATLAB documentation and references for further details.

The solver function returns a matrix with the `i`-th guess of the decision variables in the `i+1`-th column (first column contains `x0`).

A typical solver calling script would look like this:
```matlab
gs = @<stochastic gradient function>;
x0 = [0; 0; 0; 0];
nIter = 500;
idxSG = randi(<number of stochastic gradients>, 1, nIter);

xMat = <solver>(gs, x0, nIter, idxSG, <solver parameters>);
```
