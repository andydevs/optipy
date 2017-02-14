#!/usr/bin/env python

"""
Program: OptiPy

A python optimization library.

Author:  Anshul Kharbanda
Created: 8 - 11 - 2016
"""

# Unittest
import unittest

# Optipy
import optipy.helper           as opth
import optipy.gradient_descent as optgd
import optipy.newtonian        as optnt

# Numpy
import numpy
import numpy.testing as npt

class OptiPyTestCase(unittest.TestCase):
	"""
	Tests optipy library

	@author  Anshul Kharbanda
	@created 8 - 13 - 2016
	""" 
	def setUp(self):
		"""
		Sets up the unit test

		Change parameters in this function to adjust testing
		"""
		# Test function (with gradientand number of inputs)
		self.function  = lambda x: 0.5*(x[0]**2 + x[1]**2)
		self.gradient  = lambda x: x
		self.numinputs = 2

		# The critical point of the function
		self.critical = numpy.array([ 0.0, 0.0 ])

		# Set up options
		self.options = {
			'epsilon' : 1e-10,
			'delta'   : 1e-6,
			'maxiter' : 100000,
			'gfunc'   : self.gradient
		}

		# The tolerance between the calculated and actual value
		self.tol = (1 - 1e-10)

class OptiPyHelperTestCase(OptiPyTestCase):
	"""
	Tests optipy helper

	@author  Anshul Kharbanda
	@created 8 - 13 - 2016
	"""
	def setUp(self):
		"""
		Sets up the unit test
		"""
		# Call super
		super(OptiPyHelperTestCase, self).setUp()

		# Point values
		self.test_point = numpy.array([ 2.0, 2.0 ])

	def test_gradient(self):
		"""
		Tests the gradient compute method
		"""
		# Expected and modeled gradient
		self.expected = numpy.array([ 2.0, 2.0 ])
		self.modeled  = opth.gradient(self.test_point, self.function, **self.options)

		# Assert close
		npt.assert_allclose(self.expected, self.modeled, atol=self.tol)

	def test_hessian(self):
		"""
		Tests the hessian compute method
		"""
		# Expected and modeled hessian
		self.expected = numpy.array([[1.0, 0.0], [0.0, 1.0]])
		self.modeled  = opth.hessian(self.test_point, self.function,**self.options)

		# Assert close
		npt.assert_allclose(self.expected, self.modeled, atol=self.tol)

class GradientDescentTestCase(OptiPyTestCase):
	"""
	Tests gradient descent library

	@author  Anshul Kharbanda
	@created 8 - 13 - 2016
	"""
	def test_batch_descent(self):
		"""
		Test batch descent
		"""
		# Get result
		result = optgd.batch(self.function, self.numinputs, **self.options)

		# Make sure result is not nan
		self.assertFalse(numpy.isnan(result).any())

		# Result should be close to actual
		npt.assert_allclose(result, self.critical, atol=self.tol)

	def test_stochastic_descent(self):
		"""
		Test stochastic descent
		"""
		# Get result
		result = optgd.stochastic(self.function, self.numinputs, **self.options)

		# Make sure result is not nan
		self.assertFalse(numpy.isnan(result).any())

		# Result should be close to actual
		npt.assert_allclose(result, self.critical, atol=self.tol)

class NewtonianTestCase(OptiPyTestCase):
	"""
	Tests newton's method for descent

	@author  Anshul Kharbanda
	@created 8 - 13 - 2016
	"""
	def test_newtons_method(self):
		"""
		Tests newton's method
		"""
		# Get result
		result = optnt.pure(self.function, self.numinputs, **self.options)

		# Make sure result is not nan
		self.assertFalse(numpy.isnan(result).any())

		# Result should be close to actual
		npt.assert_allclose(result, self.critical, atol=self.tol)
		
# Main method
if __name__ == '__main__':
	unittest.main()