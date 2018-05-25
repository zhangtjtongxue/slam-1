"""

This file contains all kind of auxiliary functions.

Author: C.Rauterberg

"""

import numpy as np
from random import randint, choice
from measurements import get_range_bearing_measurements, normalize_angle, mean_angles
from copy import deepcopy
from math import sqrt
from scipy.linalg import inv, det, norm
from numpy.random import random
from scipy.stats import multivariate_normal as scipy_multi


def append_to_file(filename, line):
	with open(filename, "a") as myfile:
		myfile.write(line)


def comp_average_landmark_error(landmarks, predictions, eps=4.0):
	num_landmarks = len(landmarks)
	obs_landmarks = 0
	avrg_err = 0.0
	for i in range(num_landmarks):
		tmp_landmark = np.array(landmarks[i])
		tmp_pred = np.array(predictions[2*i:2*i+2])
		if abs(tmp_pred[0]) < eps and abs(tmp_pred[0]) < eps:
			continue
		avrg_err += norm(tmp_landmark - tmp_pred)
		obs_landmarks += 1
	return avrg_err/obs_landmarks


def calc_weight(x, mean, cov):
	"""
	Calculate the density for a multinormal distribution

	Taken from: https://github.com/nwang57/FastSLAM
	Thanks!
	"""
	# return scipy_multi.pdf(x, mean=mean, cov=cov)
	den = 2 * np.pi * sqrt(det(cov))
	diff = x - mean
	diff[1] = normalize_angle(diff[1])
	num = np.exp(-0.5 * dot3(diff.T, inv(cov), diff))
	result = num / den
	return result


def calc_mean(entries, index_of_angles):
	new_mean = np.mean(entries, axis=0)
	new_mean[index_of_angles] = normalize_angle(mean_angles(entries[:, index_of_angles]))
	return new_mean


def calc_diff(entries, mean, index_of_angles):
	diff = entries - mean
	for i in range(len(diff)):
		diff[i, index_of_angles] = normalize_angle(diff[i, index_of_angles])
	return diff


def most_common(lst):
	return int(max(set(lst), key=lst.count))


def stratified_resample(weights):
	"""
	Re-samples according to strafied re-sample

	Taken from: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb
	Thanks!
	:param weights:
	:return:
	"""
	N = len(weights)
	# make N subdivisions, chose a random position within each one
	positions = (random(N) + range(N)) / N

	indexes = np.zeros(N, 'i')
	cumulative_sum = np.cumsum(weights)
	i, j = 0, 0
	while i < N:
		if positions[i] < cumulative_sum[j]:
			indexes[i] = j
			i += 1
		else:
			j += 1
	return indexes


def systematic_resample(weights):
	"""
	Re-samples according to systematic re-sample

	Taken from: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb
	Thanks!
	:param weights:
	:return:
	"""
	N = len(weights)

	# make N subdivisions, choose positions
	# with a consistent random offset
	positions = (np.arange(N) + random()) / N

	indexes = np.zeros(N, 'i')
	cumulative_sum = np.cumsum(weights)
	i, j = 0, 0
	while i < N:
		if positions[i] < cumulative_sum[j]:
			indexes[i] = j
			i += 1
		else:
			j += 1
	return indexes


def generate_simulation(num_landmarks, x_min, x_max, orig_state, threshold, number_of_steps, f, Q, R, step_size=3,
						go_straight=False, do_test=False, additional_args=(1, 1), dim_state=3, do_circle=False, asso_test=False):
	"""
	Generates the data needed for the simulation: The true robots path, the controls needed and the noisy measurements needed.
	:param dim_state: The size of the state vector
	:param num_landmarks: The number of landmarks in the world
	:param x_min: Min. value in Euclidean plane
	:param x_max: Max. value in Euclidean plane
	:param orig_state: The starting position
	:param threshold: The maximum range in which to record measurements
	:param number_of_steps: The number of steps to be run in the simulation
	:param f: The state transition function, i.e. the movement model
	:param Q: The state transition covariance
	:param R: The measurement covariance
	:param step_size: The size of the steps taking a each time step t
	:param go_straight: Boolean indicating whether to go straight or take turns
	:param do_test: If true, simply place four landmarks in the world. Only for testing
	:param additional_args: Additional arguments. Must at least include arguments for the state transition function (w,delta_t)
	:return: A 5-tuple, containing:
		- The total size of the world
		- The now placed number of landmarks
		- A list of the landmarks
		- A list of the true positions of the robot in the simulation
		- A list of all the measurements recorded at each time step
		- A list of all the controls fed to the robot at each time step
	"""
	w, delta_t = additional_args[:2]

	world_size = abs(x_min) + abs(x_max)
	landmarks = [[randint(x_min + 5, x_max - 5), randint(x_min + 5, x_max - 5)] for _ in range(num_landmarks)]

	measurements_at_i = [None]
	u_at_i = [None]

	if do_test:
		landmarks = [[25, 25], [25, 75], [75, 25], [75, 75]]
		num_landmarks = len(landmarks)

	test_state = np.array([0.0 for _ in range(dim_state)]).T

	if orig_state is not None:
		test_state = deepcopy(orig_state)
	else:
		test_state[0] = (x_max - abs(x_min)) / 2
		test_state[1] = (x_max - abs(x_min)) / 2

	if do_circle:
		test_state[0] = -97.5
		test_state[1] = -5
		test_state[2] = .5 * np.pi
	if asso_test:
		test_state[0] = x_min + 5
		test_state[1] = x_max / 2
		test_state[2] = 0.0

		landmarks = [[i, 25] for i in range(x_min + 5, x_max - 5, 20)] + [[i, 75] for i in range(x_min + 5, x_max - 5, 20)]
		num_landmarks = len(landmarks)

	position_at_i = [test_state]

	current_circle_u = -0.01

	for _ in range(number_of_steps):

		if not go_straight:
			# test_u = np.array([step_size, choice([0.0, -.05, -.1, -.15, .05, .1, .15])])
			test_u = np.array([step_size, choice([0.0, -.05, -.1, .05, .1, -.025, .025])])
		else:
			test_u = np.array([step_size, 0.0])

		if do_circle:
			test_u = np.array([step_size, current_circle_u])
			current_circle_u -= 0.00001

		# check if we run into a wall
		test_u = move(test_state, test_u, x_min, x_max, f, Q, (w, delta_t))

		# do real movement - therefore no noise!
		test_state = f(test_state, test_u, Q, (w, delta_t), do_noise=False)
		test_state[2] = normalize_angle(test_state[2])

		# store current position
		position_at_i += [test_state]
		# store correct control
		u_at_i += [test_u]
		# store measurements received at i
		if do_test:
			z = get_range_bearing_measurements(test_state, landmarks, R, do_noise=True, threshold=None)
		else:
			z = get_range_bearing_measurements(test_state, landmarks, R, do_noise=True, threshold=threshold)
		measurements_at_i += [z]

	return world_size, num_landmarks, landmarks, position_at_i, measurements_at_i, u_at_i


def generate_covariance_matrices(ro_x, ro_y, ro_theta, ro_range, ro_bearing, init_ro_x, init_ro_y, init_ro_theta):
	"""
	Simple shorthandle to build the state and the measurement covariance matrix
	:param init_ro_theta: Initial standard derivation of heading for generation of ensemble/particles
	:param init_ro_y: Initial standard derivation of y for generation of ensemble/particles
	:param init_ro_x: Initial standard derivation of x for generation of ensemble/particles
	:param ro_x: The standard derivation for x in Q
	:param ro_y: The standard derivation for y in Q
	:param ro_theta: The standard derivation for theta in Q
	:param ro_range: The standard derivation for the range in R
	:param ro_bearing: The standard derivation for the bearing in R
	:return:
	"""
	Q = np.asarray([[ro_x, 0, 0], [0, ro_y, 0], [0, 0, ro_theta]])
	R = np.asarray([[ro_range, 0], [0, ro_bearing]])
	init_cov = np.asarray([[init_ro_x, 0, 0], [0, init_ro_y, 0], [0, 0, init_ro_theta]])
	return Q, R, init_cov


def generate_large_state(orig_state, num_landmarks, init_cov, dim_landmarks=2, init=True):
	new_state = np.concatenate((orig_state, [0] * (num_landmarks*dim_landmarks)))
	new_size = len(new_state)
	if init:
		new_init_cov = np.zeros((new_size, new_size))
		for i in range(init_cov.shape[0]):
			new_init_cov[i, i] = init_cov[i, i]
	else:
		new_init_cov = None
	return new_state, new_init_cov


def dot3(A, B, C):
	"""
	Just a simple short handle for multiplying three matrices
	:param A: Matrix A
	:param B: Matrix B
	:param C: Matrix C
	:return: The combined dot-product of the three
	"""
	return np.dot(np.dot(A, B), C)


def move(current_position, u, x_min, x_max, f, Q, f_args=(1, 1)):
	"""
	Check if the specified control is a legal move in the current world.
	:param current_position: The current position
	:param u: The previously given controls
	:param x_min: Minimal x-Value of the Euclidean plane
	:param x_max: Maximum x-Value of the Euclidean plane
	:param f: The movement model
	:param f_args: Additional arguments needed for the movement model
	:return: The controls u that form a legal movement
	"""
	##############################################################################################

	if len(current_position) < 3:
		raise ValueError("Current State is not large enough, should at least contain [x,y,heading]")
	if not callable(f):
		raise ValueError("Please specify the movement model as a function!")

	##############################################################################################
	new_position = f(current_position, u, Q, f_args, do_noise=False)
	robot_width = f_args[0]
	x = new_position[0]
	y = new_position[1]
	sign = -1 if u[1] < 0 else 1
	if x < (x_min + robot_width) or y < (x_min + robot_width):
		return move(current_position, [u[0], u[1] + sign * 0.01], x_min, x_max, f, Q, f_args)
	elif x >= (x_max - robot_width) or y >= (x_max - robot_width):
		return move(current_position, [u[0], u[1] + sign * 0.01], x_min, x_max, f, Q, f_args)
	else:
		return u


