"""
Program: OptiPy

A python optimization library.

Author:  Anshul Kharbanda
Created: 8 - 11 - 2016
"""

# Imports
import helper
import numpy

def batch_descent(func, lenx, **kwargs):
	"""
	Performs batch gradient descent minimization on the 
	given function with the given number of inputs

	@param func     the function to mininmize
	@param lenx     the number of inputs being passed to the function
	@param **kwargs extra arguments
		'epsilon' what is considered a small change (defaults to 1e-1)
		'step'    the step change, or learning rate (defaults to 1e-4)
		'delta'   the delta value used to calculate gradients (defaults to 1e-25)
		'maxiter' the maximum number of iterations before the program stops
					(if zero, program will not stop at max iterations. defaults to 0)
	"""
	# Get kwargs
	delta   = kwargs.get('delta',   1e-25)
	epsilon = kwargs.get('epsilon', 1e-1)
	step    = kwargs.get('step',    1e-4)
	maxiter = kwargs.get('maxiter', 0)
	
	# Initial inputs, gradients, and counter
	inputs    = numpy.random.rand(lenx) 
	gradients = numpy.zeros(lenx)
	counter   = 0

	# Until gradient is smaller than or eaual to small change
	# (or max iterations are reached)
	while helper.magnitude(gradients) > epsilon and (maxiter == 0 or counter < maxiter):
		# Batch compute gradients
		for index in xrange(lenx): 
			gradients[index] = helper.gradient(inputs, func, index, delta)

		# Step in the negative direction
		inputs -= gradients*step

		# Increment counter
		counter += 1

	# Return the optimized values
	return inputs

def stochastic_descent(func, lenx, **kwargs):
	"""
	Performs stochastic gradient descent minimization on the 
	given function with the given number of inputs

	@param func     the function to mininmize
	@param lenx     the number of inputs being passed to the function
	@param **kwargs extra arguments
		'epsilon' what is considered a small change (defaults to 1e-1)
		'step'    the step change, or learning rate (defaults to 1e-4)
		'delta'   the delta value used to calculate gradients (defaults to 1e-25)
		'maxiter' the maximum number of iterations before the program stops
					(if zero, program will not stop at max iterations. defaults to 0)
	"""
	# Get kwargs
	delta   = kwargs.get('delta',   1e-25)
	epsilon = kwargs.get('epsilon', 1e-1)
	step    = kwargs.get('step',    1e-4)
	maxiter = kwargs.get('maxiter', 0)
	
	# Initial inputs and gradients (and counter)
	inputs    = numpy.random.rand(lenx)
	gradients = numpy.zeros(lenx)
	counter   = 0

	# Until gradient is smaller than or eaual to small change
	while helper.magnitude(gradients) > epsilon and (maxiter == 0 or counter < maxiter):
		# Compute gradient at counter
		gradients[counter % lenx] = helper.gradient(inputs, func, counter % lenx, delta)

		# Step in the negative direction
		inputs -= gradients*step

		# Increase counter
		counter += 1

	# Return the optimized values
	return inputs