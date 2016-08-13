"""
Program: OptiPy

A python optimization library.

Author:  Anshul Kharbanda
Created: 8 - 11 - 2016
"""

# Unittest
import unittest

# Optipy
import optipy.helper   as opth
import optipy.gdescent as optgd

# Numpy
import numpy
import numpy.testing as npt

class OptiPyTestCase(unittest.TestCase):
	"""
	Tests optipy library

	@author  Anshul Kharbanda
	@created 8 - 13 - 2016
	"""
	def error(self, y, yHat):
		"""
		Returns the error between estimated and 
		actual y values

		@param y    the actual y value
		@param yHat the estimated y value

		@return the error between estimated and
				actual y values
		"""
		return numpy.sum((y - yHat)**2) / (2 * y.size)

	def setUp(self):
		"""
		Sets up the unit test
		"""
		# Test function (and number of inputs)
		self.function = lambda x: 0.5*(x[0]**2 + x[1]**2)
		self.num_inputs = 2

		# The critical point of the function
		self.critical = numpy.array([ 0.0, 0.0 ])

		# The tolerance between the calculated and actual value
		self.tol = 1

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
		self.expected_gradient = numpy.array([ 2.0, 2.0 ])
		self.modeled_gradient  = numpy.array([
			opth.gradient(self.test_point, self.function, 0),
			opth.gradient(self.test_point, self.function, 1)
		])

		# Assert close
		npt.assert_allclose(self.expected_gradient, self.modeled_gradient)

	def test_magnitude(self):
		"""
		Test magnitude method
		"""
		# Expected and modeled magnitude
		self.expected_magnitude = numpy.sqrt(8.0)
		self.modeled_magnitude  = opth.magnitude(self.test_point)

		# Assert close
		npt.assert_allclose(self.expected_magnitude, self.modeled_magnitude)

class GradientDescentTestCase(OptiPyTestCase):
	"""
	Tests gradient descent library

	@author  Anshul Kharbanda
	@created 8 - 13 - 2016
	"""
	def setUp(self):
		"""
		Sets up the unit test
		"""
		# Call super
		super(GradientDescentTestCase, self).setUp()

		# Set up options
		self.options = {
			'epsilon' : 1e-10,
			'delta'   : 1e-10
		}

	def test_batch_descent(self):
		"""
		Test batch descent
		"""
		# Get result
		result = optgd.batch_descent(self.function, self.num_inputs, **self.options)

		# Make sure result is not nan
		self.assertTrue(not numpy.isnan(result).any())

		# Result should be close to actual
		npt.assert_allclose(result, self.critical, atol=self.tol)

	def test_stochastic_descent(self):
		"""
		Test stochastic descent
		"""
		# Get result
		result = optgd.stochastic_descent(self.function, self.num_inputs, **self.options)

		# Make sure result is not nan
		self.assertTrue(not numpy.isnan(result).any())

		# Result should be close to actual
		npt.assert_allclose(result, self.critical, atol=self.tol)

# Main method
if __name__ == '__main__':
	unittest.main()