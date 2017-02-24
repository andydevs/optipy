# OptiPy

An optimization library written only with numpy.

```python
optipy.minimize(lambda x: 0.5*numpy.sum(x**2), 2)
```

## Quick Start Guide

Make sure you have numpy installed first.

Define a function (using `lambda` or `def`). This function must contain one input argument (a numpy array) and return a scalar output.

import numpy

function = lambda x: 0.5*numpy.sum(x**2)

Use `optipy.minimize` to return the minimum of the function. Pass your function and the number of input arguments to minimize.

```python
import numpy
import optipy

function = lambda x: 0.5*numpy.sum(x**2)
minimum  = optipy.minimize(function, 2)

# minimum = approx. [0., 0.]
```

Likewise, use `optipy.maximize` to return the maximum of the function.

```python
import numpy
import optipy

function = lambda x: -0.5*numpy.sum(x**2)
maximum  = optipy.maximize(function, 2)

# maximum = approx. [0., 0.]
```

## Optional Parameters

You can pass additional keyword arguments to the minimize/maximize functions to control their behaviour.

```python
optipy.minimize(func, 2, epsilon=1e-11, delta=1e-3)
```

The following parameters are valid:

| Argument | Description                                  | Defaults |
|:---------|:---------------------------------------------|:---------|
| epsilon  | Zero slope threshold                         | 1e-10    |
| delta    | Delta value used to compute gradient/hessian | 1e-6     |
| step     | Step size, or learning rate                  | 1e-4     |
| maxiter  | Maximum iterations (infinity if = 0)         | 0        |

## Optimization Algorithms

By default, optipy uses the Batch Gradient Descent algorithm, but also has other algorithms that can be used, either by calling them directly (they use the same interface as minimize and maximize), or by setting `default_optimizer` to the preferred algorithm to be used by `minimize/maximize`.

```python
optipy.default_optimizer = optipy.gradient_descent.stochastic
```

Here are the available algorithms:

| Function                           | Description                                                  |
|:-----------------------------------|:-------------------------------------------------------------|
| optipy.gradient_descent.batch      | Batch Gradient Descent                                       |
| optipy.gradient_descent.stochastic | Stochastic Gradient Descent                                  |
| optipy.newtonian.pure              | Newton's Method. Computes the full Hessian in each iteration |
