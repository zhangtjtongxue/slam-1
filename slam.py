"""

This file holds the derivations of the Kalman Filters composed in filter.py
	to adapt to the SLAM (Simultaneous Localization and Mapping) Problem as well as
	a FastSLAM, the particle filter and ExKF based approach.

Author:	C.Rauterberg

"""
import numpy as np
from filter import KalmanFilter, ParticleFilter, BaseEnsembleKalmanFilter
from measurements import normalize_angle, get_coordinate_from_range_bearing, normalize_angles, mean_angles, associate_measurements
from math import sqrt, atan2, cos, sin
from scipy.linalg import inv
from util import dot3, calc_weight, calc_mean, calc_diff, most_common
from numpy.random import multivariate_normal, normal
from numpy.linalg import norm as eucl_norm


class ExKFSlamKA(KalmanFilter):
	"""
	Implements a Extended Kalman Filter based SLAM approach with Known Data Association
	"""

	def __init__(self, state, sigma, dim_state, dim_z, num_landmarks, Q, R, dim_landmarks=2):
		super().__init__(state, sigma, dim_state, dim_z, Q, R)
		self.num_landmarks = num_landmarks
		self.dim_state_without_landmarks = self.dim_state - (self.num_landmarks * dim_landmarks)

	def predict(self, f, F, u=0, f_args=(), F_args=()):
		F_x = np.zeros((self.dim_state_without_landmarks, self.dim_state))
		for i in range(self.dim_state_without_landmarks):
			F_x[i, i] = 1
		self.state = f(self.state, u, self.Q, f_args, do_noise=True)
		if callable(F):
			F_mat = F(self.state, u, F_x, F_args)
		else:
			F_mat = F
		large_noise = np.matmul(np.matmul(np.transpose(F_x), self.Q), F_x)
		self.sigma = np.add(np.matmul(np.matmul(F_mat, self.sigma), np.transpose(F_mat)), large_noise)

	def update(self, z, ind):
		# check if landmark was never seen before
		position_of_landmark_in_state_vector = 2 * ind + self.dim_state_without_landmarks

		if self.state[position_of_landmark_in_state_vector] == 0.0 and self.state[
			position_of_landmark_in_state_vector + 1] == 0.0:
			# remember landmark
			self.state[position_of_landmark_in_state_vector] = self.state[0] + z[0] * cos(
				z[1] + self.state[2])
			self.state[position_of_landmark_in_state_vector + 1] = self.state[1] + z[0] * sin(
				z[1] + self.state[2])

		# compute delta - the estimated position of the landmark
		delta = np.asarray([0.0, 0.0])
		delta[0] = self.state[position_of_landmark_in_state_vector] - self.state[0]
		delta[1] = self.state[position_of_landmark_in_state_vector + 1] - self.state[1]
		# compute q
		q = np.matmul(np.transpose(delta), delta)
		# compute z_dash
		z_dash = np.array([sqrt(q), normalize_angle(atan2(delta[1], delta[0]) - self.state[2])])

		# define F_x_i
		F_x_i = np.zeros((self.dim_state_without_landmarks + self.dim_z, self.dim_state))
		for kk in range(self.dim_state_without_landmarks):
			F_x_i[kk, kk] = 1.0
		for kk in range(self.dim_z):
			F_x_i[self.dim_state_without_landmarks + kk, self.dim_state_without_landmarks + 2 * ind + kk] = 1

		# get matrix H
		H_i = np.array([[-sqrt(q) * delta[0], -sqrt(q) * delta[1], 0.0, sqrt(q) * delta[0], sqrt(q) * delta[1]],
						[delta[1], -delta[0], -q, -delta[1], delta[0]]], dtype=np.double)

		H_i = (1 / q) * np.matmul(H_i, F_x_i)

		# compute innovation
		S_k = np.add(np.matmul(np.matmul(H_i, self.sigma), np.transpose(H_i)), self.R)

		# compute Kalman gain
		K_k = np.matmul(np.matmul(self.sigma, np.transpose(H_i)), inv(S_k))

		# compute difference between measurement and estimated measurement
		diff = z - z_dash
		diff[1] = normalize_angle(diff[1])
		y_k = np.transpose(np.asmatrix(diff))

		self.state = np.add(self.state, np.asarray(np.dot(K_k, y_k).T.tolist()[0]))
		self.sigma = np.matmul(np.subtract(np.eye(self.dim_state), np.matmul(K_k, H_i)), self.sigma)


class EnsKFSlamKA(BaseEnsembleKalmanFilter):
	"""
	Implements a Ensemble Kalman Filter based SLAM approach with Known Data Association
	"""

	def __init__(self, state, N, dim_state, num_landmarks, dim_z, Q, R, init_cov=None, dim_landmarks=2):
		super().__init__(state, N, dim_state, dim_z, Q, R, init_cov)
		self.L = num_landmarks
		self.dim_landmarks = dim_landmarks
		self.state_without_landmarks = dim_state - dim_landmarks * num_landmarks
		self.eps = 4.0

	def predict(self, f, u=0, f_args=(), landmark_white_noise=1.0):
		"""
		Predict the next prior.
		:param f: The state transition function
		:param u: The control arguments for the state transition function
		:param f_args: Additional arguments for the state transition function
		:param landmark_white_noise: The spread for the white noise added to each landmark
		"""
		s_landmarks = [0] * (self.dim_landmarks * self.L)
		for ind in range((self.dim_landmarks * self.L)):
			if abs(self.state[ind + self.state_without_landmarks]) < self.eps:
				s_landmarks[ind] = 0.0
			else:
				s_landmarks[ind] = landmark_white_noise

		# s = np.concatenate((self.Q.diagonal(), s_landmarks))
		s = np.concatenate(([0.0, 0.0, 0.0], s_landmarks))

		self.ensemble = np.asarray(
			[f(member, u, self.Q, f_args, do_noise=True) + np.random.normal(0, s, self.dim_state) for member in self.ensemble])

		self.normalize_ensemble_angles()

		self.state = calc_mean(self.ensemble, 2)

	def get_ith_landmark(self, ind):
		return self.state[self.state_without_landmarks + self.dim_landmarks*ind:self.state_without_landmarks + self.dim_landmarks*ind + self.dim_landmarks]

	def get_ith_landmark_from_member(self, member, ind):
		return member[self.state_without_landmarks + self.dim_landmarks*ind:self.state_without_landmarks + self.dim_landmarks*ind + self.dim_landmarks]

	def update(self, z, h, ind, h_args=(), full_ensemble=False):
		"""
		Update the current prediction using incoming measurements
		:param full_ensemble:
		:param z: The incoming measurement
		:param h: The measurement function
		:param ind: The index of the landmark according to the measurement
		:param h_args: Additional arguments for the state transition function
		"""
		# according to roth_2017, we do not need a batch update, but can iterate over the updates individually.

		# check if landmark was never seen before
		position_of_landmark_in_state_vector = 2 * ind + self.state_without_landmarks

		new_landmark = abs(self.state[position_of_landmark_in_state_vector]) < self.eps \
							and abs(self.state[position_of_landmark_in_state_vector + 1]) < self.eps

		if new_landmark:
			# remember landmark - IN EACH MEMBER!
			for i in range(self.N):

				self.ensemble[i, position_of_landmark_in_state_vector:position_of_landmark_in_state_vector + self.dim_landmarks] \
					= get_coordinate_from_range_bearing(z, self.ensemble[i])

		if full_ensemble:
			tmp_ensemble = self.ensemble
			col_indx = [col_ind for col_ind in range(self.dim_state)]
		else:
			# lets try something: select only robot state and landmark state and form new tmp state
			# row_indx -> ensemble member, col_indx -> value in ensemble member
			col_indx = np.array([v for v in range(self.state_without_landmarks)] + [b for b in range(
				position_of_landmark_in_state_vector, position_of_landmark_in_state_vector + self.dim_landmarks)])

			tmp_ensemble = self.ensemble[:, col_indx]
		tap_N = len(tmp_ensemble)

		# Use this function for using correct landmark
		# x_h = np.asarray([h(member, h_args) for member in tmp_ensemble])
		if full_ensemble:
			x_h = np.asarray([h(member, self.get_ith_landmark_from_member(member, ind)) for member in tmp_ensemble])
		else:
			x_h = np.asarray([h(member, self.get_ith_landmark_from_member(member, 0)) for member in tmp_ensemble])

		# Angles can not be "meaned" like normal values!
		# Therefore: Use own mean function

		# Calculate the mean of the chosen ensemble
		x_dash = calc_mean(tmp_ensemble, 2)
		# Calculate the expected measurement for each ensemble member
		y_dash = calc_mean(x_h, 1)

		# Tang_2015 argue, that the following has to hold: y_dash == h(np.mean(tmp_ensemble, axis=0), h_args)

		# This is the normal calculation
		X = (1.0 / sqrt(tap_N - 1)) * (calc_diff(tmp_ensemble, x_dash, 2)).T
		Y = (1.0 / sqrt(tap_N - 1)) * (calc_diff(x_h, y_dash, 1)).T

		# Follow simple calculation for computation of Kalman gain
		D = np.dot(Y, Y.T) + self.R
		K = np.dot(X, Y.T)
		K = np.dot(K, inv(D))

		v_r = np.random.multivariate_normal([0] * self.dim_z, self.R, self.N)

		for j in range(self.N):
			# self.ensemble[j, col_indx] += np.dot(K, z + v_r[j] - x_h[j])
			diff = z - x_h[j] + v_r[j]
			diff[1] = normalize_angle(diff[1])
			update = np.dot(K, diff)

			self.ensemble[j, col_indx] += update

		self.normalize_ensemble_angles()
		self.state = calc_mean(self.ensemble, 2)

	def get_sigma(self):
		"""
		Overwrite to only compute covariance upon call!
		:return: The covariance approximation C
		"""
		x_dash = calc_mean(self.ensemble, 2)
		X = (1.0 / sqrt(self.N - 1)) * (calc_diff(self.ensemble, x_dash, 2)).T
		C = np.dot(X, X.T)
		return C

	def sigma_pos(self):
		"""
		Overwrite to only compute covariance upon call!
		:return: The covariance approximation C
		"""
		x_dash = calc_mean(self.ensemble[:, :3], 2)
		X = (1.0 / sqrt(self.N - 1)) * (self.ensemble[:, :3] - x_dash).T
		C = np.dot(X, X.T)
		return C


class EnsKFSlamUA(EnsKFSlamKA):

	def __init__(self, state, N, dim_state, num_landmarks, dim_z, Q, R, init_cov=None, dim_landmarks=2, observation_threshold=10):
		super().__init__(state, N, dim_state, num_landmarks, dim_z, Q, R, init_cov=init_cov, dim_landmarks=2)
		self.observation_counter = []
		self.observation_threshold = observation_threshold

	def add_landmark(self, z):
		self.L += 1
		self.observation_counter.append(0)
		self.dim_state += 2
		tmp_ensemble = np.zeros((self.N, self.dim_state))
		for j in range(self.N):
			tmp_ensemble[j] = np.concatenate((self.ensemble[j], get_coordinate_from_range_bearing(z, self.ensemble[j])))
		self.ensemble = tmp_ensemble
		# don't forget to update the state
		self.state = calc_mean(self.ensemble, 2)

	def delete_landmark(self, ind):
		self.L -= 1
		self.dim_state -= 2
		del self.observation_counter[ind]
		# remove landmark from all ensemble members
		# print([_ for _ in range(self.state_without_landmarks + ind, self.state_without_landmarks + ind+self.dim_landmarks)])
		self.ensemble = np.delete(self.ensemble, [_ for _ in range(self.state_without_landmarks + self.dim_landmarks*ind, self.state_without_landmarks + self.dim_landmarks*ind+self.dim_landmarks)], axis=1)
		# re-calculate estimation
		self.state = calc_mean(self.ensemble, 2)

	def associate_measurements(self, z, C, H, threshold=None):
		R_inflation = [2.5, 2.5]
		asso_mat = np.zeros((self.N, len(z)))
		for i in range(self.N):
			if i == 0:
				asso_mat[i] = associate_measurements(z, self.ensemble[i], self.R, C, H, threshold=threshold, debug=False)
			else:
				asso_mat[i] = associate_measurements(z, self.ensemble[i], self.R, C, H, threshold=threshold)
		return [most_common(list(asso_mat[:, i])) for i in range(len(z))]

	def compute_oberservables(self, threshold=None):
		# TODO: Prettify. Use list comprehension
		if threshold is None:
			return [1 for _ in range(self.L)]
		else:
			observable = []
			for i in range(self.L):
				obs = 0
				for j in range(self.N):
					landmark = self.get_ith_landmark_from_member(self.ensemble[j], i)
					own_pos = self.ensemble[j][:2]
					eucld = eucl_norm(landmark - own_pos)
					if eucld <= threshold + 2.5:
						obs += 1
				if obs / self.N < 0.5:
					observable.append(0)
				else:
					observable.append(1)
			return observable

	def full_update(self, z, h, H, threshold=None):

		associations = self.associate_measurements(z, self.sigma_pos() + self.Q, H)
		# Now, compute all landmarks we should have seen
		observable = self.compute_oberservables(threshold=threshold)

		for i in range(len(observable)):
			if i in associations:
				observable[i] -= 1

		landmarks_to_delete = []

		if len(observable) != len(self.observation_counter):
			raise Exception("Counters aren't the same length :-O")
		for i in range(len(observable)):
			self.observation_counter[i] += observable[i]
			if self.observation_counter[i] >= self.observation_threshold:
				landmarks_to_delete += [i]

		for i in range(len(associations)):
			if associations[i] == -1:
				continue
			else:
				mes = z[i]
				self.update(mes, h, associations[i], full_ensemble=True)

		for i in range(len(associations)):
			if associations[i] == -1:
				self.add_landmark(z[i])

		for i in landmarks_to_delete:
			self.delete_landmark(i)


class FastSlam1(ParticleFilter):
	"""
	Implementation of a Particle Filter based SLAM approach, using Extended Kalman Filters to monitor the landmarks

	According to: https://www.ri.cmu.edu/pub_files/pub4/montemerlo_michael_2003_1/montemerlo_michael_2003_1.pdf

		When reading the paper - carefully compare the notations, as I used different, more Kalman filter like
		notations of everything!
	"""

	def __init__(self, N, state, num_landmarks, p_0, Q, R, init_args=(), init_cov_infl=99):
		super().__init__(N, state, Q, R, init_args)
		self.p_0 = p_0
		self.new_cov_infl = init_cov_infl
		self.num_landmarks = num_landmarks

		self.particles = [[self.start_states[_], num_landmarks] + [None] * (2 * num_landmarks) for _ in range(self.N)]
		# normalize the angles
		for i in range(self.N):
			self.normalize_particle_angles(i)

	def predict(self, f, u, f_args=()):
		for i in range(self.N):
			par = self.particles[i]
			par[0] = f(par[0], u, self.Q, f_args, do_noise=True)
			self.normalize_particle_angles(i)

	def slam_update(self, z, ind, h, H_wrt_l, h_inv):
		"""
		Here we are following the approach in the above mentioned thesis - therefore we are performing everything
			in one run - no need to split to two methods.

		If you are looking at this and are wondering, I strongly recommend to check out the above mentioned thesis,
			specifically p.85
		:param z: The current measurement
		:param ind: The index of the according landmark
		:param h: Measurement function
		:param H_wrt_l: Jacobian of measurement function w.r.t. the landmark
		:param h_inv: The inverse of the measurement function
		"""
		# First thing: Loop over all particles:
		for i in range(self.N):

			par = self.particles[i]

			index_in_particle = 2 + ind * 2
			# This is with known associations, wherefore we skip the loop over all potential data associations

			# Compute prediction s_dash using the state transition function f
			s_dash = par[0]

			# first, check if the landmark was previously undiscovered:
			if par[index_in_particle] is None and par[index_in_particle + 1] is None:
				# Landmark not yet discovered. Add it
				# Then, compute the landmark position and sigma
				l_mean = h_inv(par[0], z)

				par[index_in_particle] = l_mean
				par[index_in_particle + 1] = np.eye(2) * self.new_cov_infl

				self.weights[i] = self.p_0

			else:

				# Get Jacobian
				H_wrt_l_mat = H_wrt_l(s_dash, par[index_in_particle])

				# Compute the expected measurement z_dash using the measurement function h
				z_dash = h(s_dash, par[index_in_particle])
				# Compute the difference and normalize!
				z_diff = z - z_dash
				z_diff[1] = normalize_angle(z_diff[1])

				l_sigma = par[index_in_particle + 1]

				# Compute Z_t
				Z_t = self.R + dot3(H_wrt_l_mat, l_sigma, H_wrt_l_mat.T)
				# print(H_wrt_l_mat)

				# update the landmark prediction accordingly
				l_mean = par[index_in_particle]

				# compute Kalman gain
				K = dot3(l_sigma, H_wrt_l_mat.T, inv(Z_t))

				# compute new mean
				par[index_in_particle] = l_mean + np.dot(K, z_diff)
				# compute new sigma
				par[index_in_particle + 1] = np.dot((np.eye(2) - np.dot(K, H_wrt_l_mat)), l_sigma)

				# compute new particle weight
				# if self.weights[i] * (calc_weight(z, z_dash, Z_t) + 1.e-300) == 0.0:
				# 	print("PRE-ALERT: %d" % i)
				# 	print("Particle %d Pos: %f,%f - Landmark Estimate: %f,%f - Weight: %f" % (i, par[0][0], par[0][1], par[index_in_particle][0], par[index_in_particle][1], self.weights[i]))
				# 	print(self.weights[i])
				# 	print((calc_weight(z, z_dash, Z_t) + 1.e-300))
				# 	print("....")
				# if self.weights[i] == 0.0:
				# 	print("ALERT")
				self.weights[i] *= calc_weight(z, z_dash, Z_t)
				# print("[%d] Diff is: [%f,%f] - Prob is: %f" % (i, z_diff[0], z_diff[1], calc_weight(z, z_dash, Z_t)))
				# print("Particle %d Pos: %f,%f - Landmark Estimate: %f,%f - Weight: %f" % (i, par[0][0], par[0][1], par[index_in_particle][0],par[index_in_particle][1], self.weights[i]))
				# print(z, z_dash)
		# if sum(self.weights) < 1.0:
		# 	self.normalize_weights()
		self.normalize_weights()

	def get_prediction(self):
		"""
		Compute the prediction of the particles, which is the mean w.r.t. the weights of the particles
		:return:
		"""
		# gather all position predictions
		pos_predictions = []
		for i in range(self.N):
			pos_predictions += [self.particles[i][0]]
		pos_prediction = np.average(pos_predictions, axis=0, weights=self.weights)

		l_predictions = []
		for j in range(self.num_landmarks):
			tmp_predictions = []
			tmp_weights = []
			for i in range(self.N):
				if self.particles[i][2 * j + 2] is None:
					continue
				tmp_predictions += [self.particles[i][2 * j + 2]]
				tmp_weights += [self.weights[i]]
			if len(tmp_weights) > 0.0:
				l_predictions += [np.average(tmp_predictions, axis=0, weights=tmp_weights)]
		l = np.array([item for sublist in l_predictions for item in sublist])
		return list(pos_prediction) + list(l)


class FastSlam2KA(FastSlam1):
	"""
	Implementation of a Particle Filter based SLAM approach, using Extended Kalman Filters to monitor the landmarks

	According to: https://www.ri.cmu.edu/pub_files/pub4/montemerlo_michael_2003_1/montemerlo_michael_2003_1.pdf

		When reading the paper - carefully compare the notations, as I used different, more Kalman filter like
		notations of everything!
	"""
	def __init__(self, N, state, num_landmarks, p_0, Q, R, init_args=(), init_cov_infl=99):
		super().__init__(N, state, 0, p_0, Q, R, init_args)
		self.p_0 = p_0
		self.new_cov_infl = init_cov_infl
		self.num_landmarks = num_landmarks

		self.particles = [[self.start_states[_], 0] + [None] * (2 * num_landmarks) for _ in range(self.N)]
		# normalize the angles
		for i in range(self.N):
			self.normalize_particle_angles(i)

	def get_jacobians(self, current_pos, landmark_pred, landmark_cov, h, H, H_wrt_l):
		predicted_meas = h(current_pos, landmark_pred)
		# Jacobian of h w.r.t the i-th feature
		feature_jacobian = H_wrt_l(current_pos, landmark_pred)
		# Jacobian of h w.r.t. the current state
		pose_jacobian = H(current_pos, landmark_pred)
		adj_cov = self.R + dot3(feature_jacobian, landmark_cov, feature_jacobian.T)
		return predicted_meas, feature_jacobian, pose_jacobian, adj_cov

	def slam_update(self, z, h, H, H_wrt_l, h_inv):

		for part_index in range(self.N):

			particle = self.particles[part_index]

			initial_pose = particle[0]

			pose_mean = initial_pose
			pose_cov = self.Q

			new_landmark = [False for _ in range(len(z))]

			for ind in range(len(z)):
				index_in_particle = 2 + 2 * ind
				meas = z[ind]

				if meas is None:
					continue

				current_pos = particle[0]

				# Check if we already have a value for this landmark
				if particle[index_in_particle] is None:
					new_landmark[ind] = True
					particle[index_in_particle] = h_inv(current_pos, meas)
					particle[index_in_particle + 1] = np.eye(2) * self.new_cov_infl
				else:
					# feature_jacobian denotes the Jacobian of the measurement function h
					# 		w.r.t. the i-th landmark.
					# pose_jacobian denotes the Jacobian of the measurement function h
					# 		w.r.t. the current state.
					predicted_meas, feature_jacobian, pose_jacobian, adj_cov =\
						self.get_jacobians(current_pos, particle[index_in_particle], particle[index_in_particle + 1], h, H, H_wrt_l)

					# Predict the new covariance for the state
					pose_cov = inv(dot3(pose_jacobian.T, inv(adj_cov), pose_jacobian) + inv(pose_cov))
					# Predict the new mean for the state
					pose_mean = current_pos + np.dot(dot3(pose_cov, pose_jacobian.T, inv(adj_cov)), normalize_angles(meas, predicted_meas))
					# Draw new state:
					particle[0] = multivariate_normal(mean=pose_mean, cov=pose_cov)

			# Now, after updating the pose - update the landmarks
			for ind in range(len(z)):
				index_in_particle = 2 + 2 * ind
				meas = z[ind]

				if meas is None:
					continue

				current_pos = particle[0]

				if new_landmark[ind]:
					self.weights[part_index] = self.p_0
				else:
					# feature_jacobian denotes the Jacobian of the measurement function h
					# 		w.r.t. the i-th landmark.
					# pose_jacobian denotes the Jacobian of the measurement function h
					# 		w.r.t. the current state.
					predicted_meas, feature_jacobian, pose_jacobian, adj_cov = \
						self.get_jacobians(current_pos, particle[index_in_particle], particle[index_in_particle + 1], h, H, H_wrt_l)
					# --------------------------------------------------------------------------------------
					# Update the landmark
					l_mean = particle[index_in_particle]
					l_cov = particle[index_in_particle + 1]
					# Compute Kalman gain
					K = dot3(l_cov, feature_jacobian.T, inv(adj_cov))
					# ... and update via rules from ExKF
					particle[index_in_particle] = l_mean + np.dot(K, normalize_angles(meas, predicted_meas))
					particle[index_in_particle + 1] = np.dot((np.eye(2) - np.dot(K, feature_jacobian)), l_cov)
					# --------------------------------------------------------------------------------------
					# Update the weight:
					L = dot3(pose_jacobian, self.Q, pose_jacobian.T) + dot3(feature_jacobian,
																			particle[index_in_particle + 1],
																			feature_jacobian.T) + self.R
					# print("Weight currently is %f, likelihood is %f" % (self.weights[part_index], calc_weight(meas, predicted_meas, L)))
					self.weights[part_index] *= calc_weight(meas, predicted_meas, L)

			prior = calc_weight(particle[0], initial_pose, self.Q)
			prop = calc_weight(particle[0], pose_mean, pose_cov)

			# print("We get: prior = %f, prop = %f, weight = %f" % (prior, prop, self.weights[part_index]))

			self.weights[part_index] = self.weights[part_index] * prior / prop

		# Normalize weights.
		self.weights = self.weights / sum(self.weights)
