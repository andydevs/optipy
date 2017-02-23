"""
Program: OptiPy

A python optimization library.

Author:  Anshul Kharbanda
Created: 8 - 11 - 2016
"""
import gradient_descent
import newtonian

# Default optimizer
default_optimizer = gradient_descent.batch

def minimize(func, lenx, **kwargs):
    """
    Minimizes the given function using the default minimization function

    @param func  the function to maximize
    @param lenx  the number of input arguments
    @param **kwargs extra arguments
		'epsilon' what is considered a small change (defaults to 1e-10)
		'delta'   the delta value used to calculate gradients (defaults to 1e-6)
		'maxiter' the maximum number of iterations before the program stops
					(if zero, program will not stop at max iterations. defaults to 0)
    """
    return default_optimizer(func, lenx, **kwargs)

def maximize(func, lenx, **kwargs):
    """
    Maximizes the given function using the default minimization function

    @param func  the function to maximize
    @param lenx  the number of input arguments
    @param **kwargs extra arguments
		'epsilon' what is considered a small change (defaults to 1e-10)
		'delta'   the delta value used to calculate gradients (defaults to 1e-6)
		'maxiter' the maximum number of iterations before the program stops
					(if zero, program will not stop at max iterations. defaults to 0)
    """
    return default_optimizer(lambda x: -1*func(x), lenx, **kwargs)
