"""

This file contains some small unit tests to ensure the correct working of the movement model, the according
	functions and - most of all - the correct results for the headings!

Author: C.Rauterberg

"""

import unittest
from functions import *
import numpy as np
from measurements import get_range_bearing_measurements


class FunctionTest(unittest.TestCase):

	def test_f(self):
		"""
		Test of the movement model works correctly!
		Take of those headings!
		"""
		# Test the state transition function f
		Q = np.eye(3)
		(w, delta_t) = (1, 1)
		tol = 5

		# Test of going straight works
		test_state = np.array([0.0, 0.0, 0.0])
		test_u = np.array([5.0, 0.0])
		np.testing.assert_array_almost_equal(np.array([5.0, 0.0, 0.0]), f(test_state, test_u, Q, args=(w, delta_t), do_noise=False))

		test_state = np.array([0.0, 0.0, .5*np.pi])
		test_u = np.array([5.0, 0.0])
		np.testing.assert_array_almost_equal(np.array([0.0, 5.0, .5*np.pi]), f(test_state, test_u, Q, args=(w, delta_t), do_noise=False), decimal=tol)

		test_state = np.array([0.0, 0.0, -.5 * np.pi])
		test_u = np.array([5.0, 0.0])
		np.testing.assert_array_almost_equal(np.array([0.0, -5.0, -.5 * np.pi]), f(test_state, test_u, Q, args=(w, delta_t), do_noise=False), decimal=tol)

		test_state = np.array([0.0, 0.0, -np.pi])
		test_u = np.array([5.0, 0.0])
		np.testing.assert_array_almost_equal(np.array([-5.0, 0.0, -np.pi]), f(test_state, test_u, Q, args=(w, delta_t), do_noise=False), decimal=tol)

		# Test if turning works
		test_state = np.array([0.0, 0.0, .25 * np.pi])
		test_u = np.array([5.0, 0.25])
		tmp_res = np.array([0.6838208121969491, 4.61690500201218, 2.0621077695026298])
		np.testing.assert_array_almost_equal(tmp_res, f(test_state, test_u, Q, args=(w, delta_t), do_noise=False), decimal=tol)

		# Test if turning works with critical heading angles
		test_state = np.array([0.0, 0.0, 0.9 * np.pi])
		test_u = np.array([5.0, 0.25 * np.pi])
		tmp_res = np.array([0.6906306216467092, -0.9776016435734084, 1.5442480810512267])
		np.testing.assert_array_almost_equal(tmp_res, f(test_state, test_u, Q, args=(w, delta_t), do_noise=False), decimal=tol)

		# Test if turning works with critical heading angles
		test_state = np.array([0.0, 0.0, -0.9 * np.pi])
		test_u = np.array([5.0, -0.25 * np.pi])
		tmp_res = np.array([0.6906306216467092, 0.9776016435734082, -1.5442480810512267])
		np.testing.assert_array_almost_equal(tmp_res, f(test_state, test_u, Q, args=(w, delta_t), do_noise=False), decimal=tol)

	def test_h(self):
		"""
		Check if the measurement model works correctly.
		As long as everything works out with the heading, this can be automatized.
		"""
		# test the measurement function h
		landmarks = [[10, 10], [-10, 10], [10, -10], [-10, -10]]
		headings = [.25*np.pi, .75*np.pi, -.25*np.pi, -.75*np.pi]

		tol = 5
		R = np.eye(2)

		for l in landmarks:
			for head in headings:
				test_state = np.array([0.0, 0.0, head])
				z = get_range_bearing_measurements(test_state, [l], R, do_noise=False)[0]
				z_dash = h(test_state, l)
				np.testing.assert_array_almost_equal(z, z_dash, decimal=tol)
				l_inv = h_inv(test_state, z)
				np.testing.assert_array_almost_equal(l, l_inv, decimal=tol)


if __name__ == '__main__':
	unittest.main()
