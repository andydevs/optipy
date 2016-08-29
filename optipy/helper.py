"""
Program: OptiPy

A python optimization library.

Author:  Anshul Kharbanda
Created: 8 - 11 - 2016
"""

# Modules
import numpy

def gradient(input, func, index=None, **kwargs):
	"""
	Returns the gradient of the f at the given input i

	@param input the array of input values to optimize
	@param func the function to compute the output
	@param index the index to compute the gradient of
	@param kwargs extra arguments
		'delta' : the gradient change value

	@return the gradient of the f at the given input i
	"""
	# Get kwargs
	delta = kwargs.get('delta', 1e-10)

	if index is None:
		return numpy.array([
			gradient(input, func, jndex)
				for jndex in xrange(len(input))
		])
	else:
		# Compute the delta vector
		delta_x = numpy.zeros(len(input))
		delta_x[index] = delta

		# Return gradient approximation at delta
		return (func(input + delta_x) - func(input)) / delta