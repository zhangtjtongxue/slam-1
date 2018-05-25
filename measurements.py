"""

This file contains all kind of auxiliary functions used for the computations of
	measurements as well as data association.

Author: C.Rauterberg

"""

from scipy.optimize import linear_sum_assignment
from math import sqrt, atan2, cos, sin
import numpy as np
from copy import deepcopy

NEW_LANDMARK = -1


def get_range_bearing_measurements(current_state, landmarks, R, do_noise=True, threshold=None, dim_z=2):
	"""
	Creates measurements containing (range, bearing) to all landmarks
	:param R: The matrix defining the errors for the sensors on its diagonal
	:param do_noise: Boolean value indicating whether to incorporate noise into measurements or not
	:param threshold: Boolean value indicating whether to cut measurements of from a certain distance on or not
	:param current_state: The current position of the robot
	:param landmarks: The positions of all the landmarks
	:param dim_z: The dimension of the measurement
	:return:
	"""
	##############################################################################################

	if len(current_state) < 3:
		raise ValueError("Current State is not large enough, should at least contain [x,y,heading]")
	if R.shape[0] != dim_z or R.shape[1] != dim_z:
		raise ValueError("Measurement Covariance Matrix R is not of size (%dx%d)" % (dim_z, dim_z))

	##############################################################################################

	z = []
	x, y, heading = current_state[0], current_state[1], current_state[2]

	for l in landmarks:
		d = sqrt((l[0] - x) ** 2 + (l[1] - y) ** 2)
		if threshold is not None and d > threshold:
			z += [None]
		else:
			if do_noise:
				noise = np.random.normal(0, R.diagonal(), dim_z)
				z += [np.array([d + noise[0], normalize_angle(atan2((l[1] - y), (l[0] - x)) - heading + noise[1])])]
			else:
				z += [np.array([d, normalize_angle(atan2((l[1] - y), (l[0] - x)) - heading)])]

	z = np.asarray([elem for elem in z])
	return z


def get_coordinate_from_range_bearing(measurement, state):
	"""
	Computes the coordinates of a landmark relative to a given position provided a measurement from this position
	:param measurement: The measurement of the form [ range, bearing ]
	:param state: The current state of the form [ x, y, heading, ... ]
	:return:
	"""
	r, b = measurement
	x, y, heading = state[:3]
	angle = b + heading
	angle = normalize_angle(angle)
	return np.array([x + r * cos(angle), y + r * sin(angle)])


def normalize_angle(angle):
	"""
	Normalizes the angle to be between [-pi, pi)
	:param angle: The angle to be normalized
	:return: The angle normalized to [-pi, pi)
	"""
	angle = deepcopy(angle) % (2 * np.pi)
	if angle > np.pi:
		angle -= 2 * np.pi
	return angle


def normalize_angles(sub_1, sub_2):
	diff = sub_1 - sub_2
	if len(diff) == 2:
		diff[1] = normalize_angle(diff[1])
	else:
		for i in range(len(diff)):
			diff[i][1] = normalize_angle(diff[i][1])
	return diff


def mean_angles(angles):
	sin_sum = sum([sin(alpha) for alpha in angles])
	cos_sum = sum([cos(alpha) for alpha in angles])
	return atan2(sin_sum, cos_sum)


def permute_measurements(z):
	"""
	Shuffles a given array.
	:param z: The array to be shuffled
	:return: A shuffled copy of the array
	"""
	tmp = np.array(deepcopy(z))
	np.random.shuffle(tmp)
	indices = [np.where(tmp == true_meas)[0][0] for true_meas in z]
	return tmp, indices


def calc_diff_meas(entry_one, entry_two, index_of_angles):
	diff = entry_one - entry_two
	diff[index_of_angles] = normalize_angle(diff[index_of_angles])
	return diff


def associate_measurements(z, current_state, R, C, H, s=5.991, threshold=None, dim_robot_state=3, dim_measurements=2, debug=False):
	"""
	This method will associate a list of given measurements to the landmarks provided in the given state vector
	:param z: The list of measurements
	:param current_state: The current state containing all previously seen landmarks
	:param R: The error matrix for the measurement function
	:param s: Threshold for defining new measurements. Defaults to s=5.991, the 95%-value from the chi-square distribution
	:param threshold: The distance threshold for the range sensor
	:param dim_robot_state: The length of the true robot state  in the state vector. Defaults to 3.
	:param dim_measurements: The length of a measurement. Defaults to 2.
	:return: A 4-tuple, containing:
		- The computed (x,y) coordinates from a measurement
		- The (x,y) coordinates of the according landmark
		- The (x,y) coordinates of all unobserved landmarks
		- The (x,y) coordinates of all NEW landmarks
	"""
	# pull landmarks from state for convenience
	landmarks = np.array([[current_state[dim_robot_state + 2 * i], current_state[dim_robot_state + 2 * i + 1]] for i in range(int((len(current_state) - dim_robot_state) / 2))])

	# get the expected range/bearing measurements we would expect for each landmark
	correct_z = get_range_bearing_measurements(current_state, landmarks, R, do_noise=False, threshold=threshold)

	# first, lets try to find out if we have new measurements!
	assignment_matrix = np.zeros((len(correct_z), len(z)))
	for i in range(len(correct_z)):
		for j in range(len(z)):
			dif = calc_diff_meas(correct_z[i], z[j], 1)
			# dif = correct_z[i] - z[j]
			H_mat = H(current_state, landmarks[i])
			sigma = np.dot(np.dot(H_mat, C), H_mat.T) + R
			assignment_matrix[i, j] = (dif[0] / R[0, 0]) ** 2 + (dif[1] / R[1, 1]) ** 2

	if debug:
		print(assignment_matrix)
	# to remove:
	# all columns represent a measurement, that is not normally distributed to any previously observed landmark
	#  --> Store as new landmarks

	# all rows represent a landmark, that has no measurement that appears to be normally distributed to it.
	#  --> Discard
	if len(assignment_matrix) == 0:
		# This only happens if there are no (!) landmark in state --> all landmarks are new landmarks
		return [-1 for _ in range(len(z))]

	new_measurements_indices = np.where(assignment_matrix.min(0) >= s*2)[0]
	# we can ignore those for the end result.
	unobserved_landmark_indices = np.where(assignment_matrix.min(1) >= s*2)[0]

	# split between new measurements and measurements of previously seen landmarks.
	remaining_measurement_indices = list(set(range(len(z))) - set(new_measurements_indices))
	remaining_measurements = z[remaining_measurement_indices]

	if debug:
		print("New measurement indices: ", new_measurements_indices)
		print("Unobserved landmark indices: ", unobserved_landmark_indices)

	# observed_landmarks for calculating the distance for the Hungarian method
	possible_observed_landmark_indices = np.array(list(set(range(len(landmarks))) - set(unobserved_landmark_indices)))
	possible_observed_landmarks = landmarks[list(set(range(len(landmarks))) - set(unobserved_landmark_indices))]

	# work with mixed measurement
	# we must achieve the same space for computation of the norm
	# map each measurement to real world coordinate
	measurements_to_world_coordinates = []
	for meas in remaining_measurements:
		measurements_to_world_coordinates += [get_coordinate_from_range_bearing(meas, current_state)]

	# Reserve the assignment matrix
	assignment_matrix = np.zeros((len(possible_observed_landmarks), len(remaining_measurements)))

	for i in range(len(possible_observed_landmarks)):
		for j in range(len(remaining_measurements)):
			assignment_matrix[i, j] = np.linalg.norm(possible_observed_landmarks[i, :] - measurements_to_world_coordinates[j])

	row_ind, col_ind = linear_sum_assignment(assignment_matrix)

	if debug:
		print(assignment_matrix)

	# Situation we might face now is, that we have k measurements of previously seen landmarks
	# and still have p landmarks, where k <= p
	# if k = p: easy, matching done
	# if k < p: This means, that we have more landmarks than measurements, but there is measurement in the 95%
	# distribution of this landmark

	# The hungarian method will make sure, that we obtain the correct/best matching

	# col_ind holds the matching for all best matchings - might be, that
	#  we have encoutered a match, but there might be a better obe:
	#  [[    1.878   979.394    84.182   954.302]
	#  [   74.186  1937.992     0.049  1883.296]
	#  [ 1145.91      1.307  1762.903     5.821]]
	# This will give use [0,1,2,3], but we clearly need to remove the col_ind first!
	# as col_ind = [0 2 1] - and these are the best matches.

	# assigned_observed_landmarks = possible_observed_landmark_indices[col_ind]

	# assigned_indices = col_ind
	# remaining_measurement_indices = list(set(remaining_measurement_indices) - set(assigned_indices))

	final_assignment_indices = [0 for _ in range(len(z))]

	if debug:
		print(possible_observed_landmark_indices)
		print("Col_ind: ", col_ind)
		print("row_ind: ", row_ind)
		print(remaining_measurement_indices)
		print("-----")

	# loop over all new measurements:
	for i in new_measurements_indices:
		final_assignment_indices[i] = NEW_LANDMARK
	for i in range(len(col_ind)):
		final_assignment_indices[remaining_measurement_indices[col_ind[i]]] = possible_observed_landmark_indices[row_ind[i]]

	# Define unused measurements as new measurements
	last_new_measurements = list(set([kk for kk in range(len(remaining_measurement_indices))]) - set(col_ind))

	for i in range(len(last_new_measurements)):
		final_assignment_indices[remaining_measurement_indices[last_new_measurements[i]]] = NEW_LANDMARK


	return np.array(final_assignment_indices)


