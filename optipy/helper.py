"""
Program: OptiPy

A python optimization library.

Author:  Anshul Kharbanda
Created: 8 - 11 - 2016
"""

# Modules
import numpy as np

def gradient(input, func, index=None, **kwargs):
	"""
	Returns the gradient of the function at the given input index
	(or all gradients if index is not given)

	@param input  the array of input values to optimize
	@param func   the function to compute the output
	@param index  the index to compute the gradient of (optional)
	@param kwargs extra arguments
		'delta' : the gradient change value
		'gfunc' : a separate gradient function

	@return the gradient of the function at the given input index
			(or all gradients if index is not given)
	"""
	# Get kwargs
	delta = kwargs.get('delta', 1e-10)
	gfunc = kwargs.get('gfunc', None)

	# If no index or gfunc is given
	if index is None and gfunc is None:
		# Compute all gradients
		return np.array([
			gradient(input, func, jndex)
				for jndex in xrange(len(input))
		])
	# If no index is given (but gfunc is given)
	elif index is None:
		# Compute gradient using gfunc
		return gfunc(input)
	else:
		# Compute the delta vector
		delta_x = np.zeros(len(input))
		delta_x[index] = delta

		# Return gradient approximation at delta
		return (func(input + delta_x) - func(input)) / delta

def hessian(input, func, **kwargs):
	"""
	Returns the hessian of the function at the given input

	@param input    the input at which to compute the hessian
	@param func     the function to compute the hessian of
	@param **kwargs extra arguments
		'delta' : the gradient (and hessian) change value
	"""
	# Get kwargs
	delta = kwargs.get('delta', 1e-10)

	# Create hessian matrix
	Hessian = np.zeros((input.size, input.size))

	# Create deltax vector
	deltax = np.zeros(input.size)

	# For each index of the hessian
	for index in xrange(Hessian.shape[0]):
		for jndex in xrange(Hessian.shape[1]):
			# Set delta at jndex location
			deltax[jndex] = delta

			# Compute point and change
			point  = gradient(input,          func, index, **kwargs)
			change = gradient(input + deltax, func, index, **kwargs)

			# Compute second derivative
			Hessian[index, jndex] = (change - point) / delta

			# Reset jndex location
			deltax[jndex] = 0

	# Return hessian
	return Hessian
