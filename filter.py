"""

This file holds the derivations of the Kalman Filters.

Author:	C.Rauterberg

"""

import numpy as np
from numpy.linalg import inv, det
from numpy.random import multivariate_normal, uniform, random, choice
from math import *
from util import dot3, calc_weight, stratified_resample, systematic_resample
from measurements import normalize_angle
from copy import deepcopy
from scipy.stats import norm as scipy_norm
from scipy.stats import multivariate_normal as scipy_multi


class KalmanFilter:
	"""
	Represents the base for every filter. Contains only Constructor and Getters.
	Author: C.Rauterberg
	"""

	def __init__(self, state, sigma, dim_state, dim_z, Q, R):
		"""
		Initializes the basic variables for the filter.
		:param state: The initial state of the filter
		:param sigma: The initial covariance of the filter
		:param dim_state: The dimension of the state variable
		:param dim_z: The dimension of the measurement observations
		:param Q: The process noise of the system
		:param R: The measurement/observation noise of the system
		"""
		self.state = state
		self.sigma = sigma
		self.dim_state = dim_state
		self.dim_z = dim_z
		self.Q = Q
		self.R = R

	def state(self):
		"""
		Return the current state of the filter
		:return: The current state of the filter
		"""
		return self.state

	def sigma(self):
		"""
		Return the current covariance matrix of the filter
		:return: The current covariance matrix
		"""
		return self.sigma


class ExtendedKalmanFilter(KalmanFilter):
	"""
	This class implements the standard Extended Kalman Filter.
	Author: C.Rauterberg
	"""

	def predict(self, f, F, u=0, f_args=(), F_args=(), do_noise=True):
		"""
		Runs the prediction step by shifting the state according to state transition
		:param do_noise: Indicator for noise according to Q. Only for testing.
		:param f: The state transition function
		:param F: The Jacobian of the state transition function. Can be function or matrix
		:param u: The control input
		:param f_args: Additional arguments needed for the state transition function
		:param F_args: Additional arguments needed for the function to compute the Jacobian F
		:return:
		"""
		# x_k = f(x_k-1, u_k)
		self.state = f(self.state, u, self.Q, f_args, do_noise=do_noise)
		# P_k = F_k-1 * P_k-1 * F_k-1^T + Q
		if callable(F):
			F_mat = F(self.state, u, F_args)
		else:
			F_mat = F
		self.sigma = dot3(F_mat, self.sigma, F_mat.T) + self.Q

	def update(self, z, h, H, h_args=(), H_args=()):
		# y_k holds the innovation
		y_k = z - h(self.state, h_args)
		H_mat = H(self.state, H_args)
		# S_k is the innovation covariance: S_k = H_k * P_k * H_k^T + R
		S_k = dot3(H_mat, self.sigma, H_mat.T) + self.R
		# Compute Kalman gain: K_k = P_k * H_k^T * S_k^-1
		K_k = dot3(self.sigma, H_mat.T, inv(S_k))
		self.state = self.state + np.dot(K_k, y_k)
		self.sigma = np.dot(np.eye(self.dim_state) - np.dot(K_k, H_mat), self.sigma)


class BaseEnsembleKalmanFilter(KalmanFilter):
	"""
	This class represents the base for all Ensemble Kalman Filters
	"""

	def __init__(self, state, N, dim_state, dim_z, Q, R, init_cov):
		super().__init__(state, None, dim_state, dim_z, Q, R)
		self.N = N

		# check how we want our initial ensemble to be distributed:
		# 1) If a "initial covariance" is given, draw samples based in that
		# 2) If no "Initial Covariance" is given, simply copy start vector
		if init_cov is not None:
			# This is a version, we we create a sample based on some initial covariance
			self.ensemble = np.random.multivariate_normal(state, init_cov, N)
		else:
			# This is a version where we copy each start vector
			self.ensemble = np.repeat(self.state[np.newaxis, :], N, axis=0)

		self.normalize_ensemble_angles()
		self.state = np.mean(self.ensemble, axis=0)

	def ensemble(self):
		return self.ensemble

	def normalize_ensemble_angles(self):
		"""
		Normalize an angle to be within [-pi, pi)
		"""
		for i in range(self.N):
			self.ensemble[i, 2] = normalize_angle(self.ensemble[i, 2])


class EnsembleKalmanFilter(BaseEnsembleKalmanFilter):
	"""
	This is an implementation of the Ensemble Kalman Filter using stochastic updates following the algorithm presented in
	<http://www.jonathanrstroud.com/papers/enkf-tutorial.pdf>
	"""

	def predict(self, f, u=0, f_args=()):
		# pass each ensemble member through the according measurement function
		# possible error with noise here!!!
		self.ensemble = np.asarray([f(member, u, self.Q, f_args, do_noise=True)
										+ np.random.normal(0, self.Q.diagonal(), len(member))
											for member in self.ensemble])
		self.state = np.mean(self.ensemble, axis=0)
		self.normalize_ensemble_angles()

	def update(self, z, h, h_args=()):
		x_h = np.asarray([h(member, h_args) for member in self.ensemble])
		x_dash = np.mean(self.ensemble, axis=0)
		y_dash = np.mean(x_h, axis=0)

		y_diff = x_h - y_dash
		for i in range(len(y_diff)):
			y_diff[i][1] = normalize_angle(y_diff[i][1])

		X = (1.0 / sqrt(self.N - 1)) * (self.ensemble - x_dash).T
		Y = (1.0 / sqrt(self.N - 1)) * y_diff.T

		D = np.dot(Y, Y.T) + self.R
		K = np.dot(X, Y.T)
		K = np.dot(K, inv(D))

		v_r = np.random.multivariate_normal([0] * self.dim_z, self.R, self.N)
		for j in range(self.N):
			diff = z + v_r[j] - x_h[j]
			diff[1] = normalize_angle(diff[1])
			self.ensemble[j] += np.dot(K, diff)

		self.state = np.mean(self.ensemble, axis=0)


class ParticleFilter:
	"""
	Standard approach to a particle filter for solving localization problems
	"""

	def normalize_particle_angles(self, ind):
		"""
		Normalize an angle to be within [-pi, pi)
		"""
		if len(self.particles[ind]) == 3:
			angle = self.particles[ind][2].copy() % (2 * np.pi)
			if angle > np.pi:
				angle -= (2 * np.pi)
			self.particles[ind][2] = angle
		elif len(self.particles[ind]) > 3:
			angle = self.particles[ind][0][2].copy() % (2 * np.pi)
			if angle > np.pi:
				angle -= (2 * np.pi)
			self.particles[ind][0][2] = angle

	def create_uniform_particles(self, x_range, y_range, hdg_range):
		"""
		Draws initial particles from a uniform distribution
		:param x_range: Tuple giving (x_min, x_max)
		:param y_range: Tuple giving (y_min, y_max)
		:param hdg_range: Tupe giving (heading_min, heading_max)
		:return: A list of particles
		"""
		particles = np.empty((self.N, 3))
		particles[:, 0] = uniform(x_range[0], x_range[1], size=self.N)
		particles[:, 1] = uniform(y_range[0], y_range[1], size=self.N)
		particles[:, 2] = uniform(hdg_range[0], hdg_range[1], size=self.N)
		return particles

	def create_gaussian_particles(self, mean, std):
		"""
		Draws initial particles from a normal distribution
		:param mean: The mean of the distribution
		:param std: The covariance of the distribution
		:return: A list of particles
		"""
		particles = multivariate_normal(mean, std, self.N)
		return particles

	def __init__(self, N, state, Q, R, init_args=()):
		self.N = N
		self.state = state
		self.Q = Q
		self.R = R
		self.weights = [1.0 for _ in range(self.N)]
		if init_args[0] == 'gaussian':
			start_states = self.create_gaussian_particles(mean=self.state, std=init_args[1])
		elif init_args[0] == 'uniform':
			start_states = self.create_uniform_particles(x_range=init_args[1], y_range=init_args[2],
														hdg_range=init_args[3])
		elif init_args[0] == 'repeat':
			s = np.asarray(state)
			start_states = np.repeat(s[np.newaxis, :], N, axis=0)
		else:
			raise ValueError("No particle distribution method given")
		self.start_states = start_states
		self.particles = [start_states[_] for _ in range(self.N)]
		# normalize the angles
		for i in range(self.N):
			self.normalize_particle_angles(i)

	def predict(self, u, f, f_args=()):
		"""
		Predict the next step - transform all particles using movement model
		:param u: The control input
		:param f: The state transition function
		:param f_args: Additional arguments needed for state transition function
		"""
		for i in range(self.N):
			part = self.particles[i]
			noise = np.random.normal([0.0], self.Q.diagonal(), len(part))
			self.particles[i] = f(part, u, self.Q, f_args) + noise
			self.normalize_particle_angles(i)

	def update(self, z, h, according_landmark):
		"""
		Update the weights of each particle, in-cooperating the given measurement
		:param z: The measurement
		:param h: The measurement function
		:param according_landmark: The according landmark position
		"""
		# weights = [1.0 for _ in range(self.N)]
		for i in range(self.N):
			expected = h(self.particles[i], according_landmark)
			self.weights[i] *= calc_weight(x=z, mean=expected, cov=self.R)
		self.normalize_weights()
		# self.weights = weights

	# "Both systematic and stratified perform very well. Systematic sampling does an excellent job
	#  of ensuring we sample from all parts of the particle space while ensuring larger weights are
	#  proportionality resampled more often. Stratified resampling is not quite as uniform as systematic
	#  resampling, but it is a bit better at ensuring the higher weights get resampled more." - R. Labbe
	# 		--> See link in function description

	def resample(self):
		"""
		Resample the particles
		"""
		new_particles = []
		new_weights = []
		# indices = systematic_resample(self.weights)
		indices = stratified_resample(self.weights)
		np_indices = sorted(choice([_ for _ in range(len(self.weights))], p=self.weights, size=self.N))

		indices = np_indices
		for j in indices:
			new_particle = deepcopy(self.particles[j])
			new_particles.append(new_particle)
			new_weights.append(self.weights[j])

		self.particles = new_particles
		self.weights = new_weights
		self.normalize_weights()

	def normalize_weights(self):
		s = np.asarray(self.weights) / (sum(self.weights))
		self.weights = s

	def get_prediction(self):
		"""
		Compute the prediction of the particles, which is the mean w.r.t. the weights of the particles
		:return:
		"""
		# gather all position predictions
		pos_predictions = []
		for i in range(self.N):
			pos_predictions += [self.particles[i]]
		if sum(self.weights) == 0.0:
			pos_prediction = np.average(pos_predictions, axis=0)
		else:
			pos_prediction = np.average(pos_predictions, axis=0, weights=self.weights)
		return pos_prediction





