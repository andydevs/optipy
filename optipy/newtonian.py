"""
Program: OptiPy

A python optimization library.

Author:  Anshul Kharbanda
Created: 8 - 11 - 2016
"""

# Imports
import sys
import helper
import numpy as np
import numpy.random as rd
import numpy.linalg as lg

def pure(func, lenx, **kwargs):
	"""
	Performs newton's method for minimization on the given function with the
	given number of inputs, computing pure Hessian

	@param func     the function to mininmize
	@param lenx     the number of inputs being passed to the function
	@param **kwargs extra arguments
		'epsilon' what is considered a small change (defaults to 1e-10)
		'delta'   the delta value used to calculate gradients (defaults to 1e-6)
		'maxiter' the maximum number of iterations before the program stops (defaults to sys.maxint)
	"""
	# Get kwargs
	epsilon = kwargs.get('epsilon', 1e-10)
	maxiter = kwargs.get('maxiter', sys.maxint)

	# Starting values
	counter  = 0
	inputs   = rd.rand(lenx)
	shift    = (1 + epsilon)*np.ones(lenx)

	# Until change is less than epsilon
	while lg.norm(shift) > epsilon and (maxiter == 0 or counter < maxiter):
		# Compute gradient and hessian
		gradient = helper.gradient(inputs, func, **kwargs)
		hessian  = helper.hessian(inputs, func, **kwargs)

		# Compute shift vector
		shift = np.dot( gradient, lg.inv( hessian ) )

		# Shift inputs
		inputs -= shift

		# Increment counter
		counter += 1

	# Return optimized inputs
	return inputs
