"""
Program: OptiPy

A python optimization library.

Author:  Anshul Kharbanda
Created: 8 - 11 - 2016
"""

# Modules
import numpy

def gradient(input, func, index, delta=1e-10):
	"""
	Returns the gradient of the f at the given input i

	@param input the array of input values to optimize
	@param func the function to compute the output
	@param index the index to compute the gradient of
	@param delta the gradient change value

	@return the gradient of the f at the given input i
	"""
	# Compute the delta vector
	delta_x = numpy.zeros(len(input))
	delta_x[index] = delta

	# Return gradient approximation at delta
	return (func(input + delta_x) - func(input)) / delta

def magnitude(vector):
	"""
	Returns the magnitude of the given vevtor

	@param vector the vector to return the magnitude of

	@return the magnitude of the given vector
	"""
	return numpy.sqrt(numpy.sum(vector**2))