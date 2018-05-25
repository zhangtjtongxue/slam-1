"""

This file contains state transition and measurement related functions for our filters.

Author: C.Rauterberg

"""

from math import *
import numpy as np
from measurements import normalize_angle, get_range_bearing_measurements, get_coordinate_from_range_bearing


def f(state, u, Q, args, do_noise=False, dim_robot_state=3):
	"""
	The underlying state transition model for our robot.

	It is very well explained in: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/10-Unscented-Kalman-Filter.ipynb

	:param dim_robot_state: The dimension of the robot state. Defaults to | [x,y,heading] | = 3
	:param state: The current state
	:param u: The control input u = (v,alpha)
	:param Q: The state transition covariance matrix
	:param args: Additional arguments, carry at least the robots width and the time difference delta_t
	:param do_noise: Boolean indicating whether to add noise or to do true movement
	:return: The new state
	"""
	##############################################################################################

	if len(state) < 3:
		raise ValueError("Current State is not large enough, should at least contain [x,y,heading]")
	if len(args) < 2:
		raise ValueError("The robot width and the time step must be specified in args!")
	if len(u) != 2:
		raise ValueError("Please specify u = (v, alpha)")

	##############################################################################################

	# Get the robots width
	w = args[0]
	# Get the time change
	delta_t = args[1]
	# Compute the distance travelled based on velocity and time based
	if do_noise and len(Q) == len(u):
		noise = np.random.normal(0, Q.diagonal(), len(u))
		velocity = u[0] + noise[0]
		alpha = u[1] + noise[1]
	else:
		velocity = u[0]
		alpha = u[1]
	d = velocity * delta_t
	# Compute the additional turning for the heading
	beta = (d / w) * tan(alpha)
	theta = state[2]
	new_state = np.transpose(np.asarray(np.array([0.0 for _ in range(len(state))])))

	if abs(u[1]) > 0.001:
		R = d / beta
		angle_sum = normalize_angle(theta + beta)
		arg = np.array(	[-R * sin(theta) + R * sin(angle_sum),
						R * cos(theta) - R * cos(angle_sum),
						beta])
	else:
		arg = np.array([d * cos(theta),
						d * sin(theta),
						0])
	if do_noise and len(Q) == dim_robot_state:
		noise = np.random.normal(0, Q.diagonal(), dim_robot_state)
		arg += noise
	for i in range(len(arg)):
		new_state[i] = arg[i]
	new_state[2] = normalize_angle(arg[2])
	res = np.add(state, new_state)
	return res


def F(state, u, args):
	"""
	Computes the Jacobian for the transition function f.
	:param state: The current state of the robot
	:param u: The control input
	:param args: Additional arguments, here used to carry omega and delta_t
	:return: The Jacobian F of the transition function f.
	"""
	w = args[0]
	delta_t = args[1]
	d = u[0] * delta_t
	beta = (d / w) * tan(u[1])
	theta = state[2]
	if abs(u[1]) > 0.001:
		R = d / beta
		return np.array([[1,	 0,	 -R * cos(theta) + R * cos(theta + beta)],
						[0,		 1,	 -R * sin(theta) + R * sin(theta + beta)],
						[0,		 0,	 1]])
	else:
		return np.array([[1,	 0,	 -d * sin(theta)],
						[0,		 1,	 d * cos(theta)],
						[0,		 0,	 1]])


def h(state, m_i):
	"""
	The measurement function, translating the current state of the robot/filter into measurement space
	:param state: The current state
	:param m_i: The position of the landmark corresponding to the measurement
	:return: The expected (r,b) measurement
	"""
	l_x, l_y = m_i[0], m_i[1]
	x, y, heading = state[0], state[1], state[2]
	angle = normalize_angle(atan2((l_y - y), (l_x - x)) - heading)
	return np.array([sqrt((l_x - x) ** 2 + (l_y - y) ** 2), angle])


def h_inv(state, z_i):
	"""
	Simple short-handle. The inverse of the measurement function computes the coordinates of a landmark
		given a measurement relative to the current state.
	:param state: The current state
	:param z_i: The measurement corresponding to the desired landmark
	:return: The [x,y] coordinates of the landmark
	"""
	return get_coordinate_from_range_bearing(z_i, state)


def H(state, m_i):
	"""
	Computes the Jacobian of the measurement function h.
	This is w.r.t to the current state, giving us a 3x2 matrix!
	:param state: The current state
	:param m_i: The position of the landmark corresponding to the measurement
	:return: The Jacobian H of the measurement function h w.r.t. to the state
	"""
	p_x, p_y = m_i[0], m_i[1]
	x, y, heading = state[0], state[1], state[2]
	q = (p_x - x) ** 2 + (p_y - y) ** 2
	return np.array([[(-p_x + x) / sqrt(q),	 (-p_y + y) / sqrt(q),	 0],
					[(-p_y + y) / q,		 (p_x - x) / q,			 -1]])


def H_wrt_l(rob_pos, l_pos):
	"""
	Computes the Jacobian of the measurement function h.
	This is w.r.t to the landmark l, giving us a 2x2 matrix!
	:param rob_pos: The current position of the robot
	:param l_pos: The current position of the landmark
	:return: The Jacobian H of the measurement function h w.r.t. the landmark l
	"""
	p_x, p_y = rob_pos[0], rob_pos[1]
	l_x, l_y = l_pos[0], l_pos[1]
	q = (l_x - p_x) ** 2 + (l_y - p_y) ** 2
	return np.array([[(l_x - p_x) / sqrt(q),	 (l_y - p_y) / sqrt(q)],
					[(-(l_y - p_y)) / q,		 (l_x - p_x) / q]])


def F_slam(state, u, F_x, args):
	w = args[0]
	delta_t = args[1]
	d = u[0] * delta_t
	beta = (d / w) * tan(u[1])
	theta = state[2]
	if abs(u[1]) > 0.001:
		R = d / beta
		tmp = np.array([[1, 0, -R * cos(theta) + R * cos(theta + beta)], [0, 1, -R * sin(theta) + R * sin(theta + beta)],[0, 0, 1]])
	else:
		tmp = np.array([[1, 0, -d * sin(theta)], [0, 1, d * cos(theta)], [0, 0, 1]])
	return np.add(np.eye(F_x.shape[1]), np.matmul(np.matmul(F_x.T, tmp), F_x))
