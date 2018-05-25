"""

TODO

Author:	C.Rauterberg

"""

import numpy as np
from slam import ExKFSlamKA, EnsKFSlamKA, FastSlam1, FastSlam2KA, EnsKFSlamUA
from functions import h, H, f, F, H_wrt_l, h_inv, F_slam
from util import generate_simulation, generate_covariance_matrices, generate_large_state, comp_average_landmark_error, append_to_file
from viz import plot_final_slam_result, visualize_position_error_history, plot_victoria
from oct2py import octave
from time import time
from copy import deepcopy
from scipy.linalg import inv, det, norm
from measurements import get_coordinate_from_range_bearing, mean_angles, normalize_angle
import os
from math import radians, sin, cos, tan
from scipy.io import loadmat


def compute_measurements(laser_scanner, thresh = 25):
	dist, angl, num = [], [], 0.0
	features = []
	for k in range(len(laser_scanner)):
		if laser_scanner[k] < 8000:
			c_l = float(laser_scanner[k])
			if len(dist) == 0 or abs(c_l - dist[-1]) < thresh:
				dist += [c_l]
				angl += [normalize_angle(radians(k/2) - np.pi/2)]
				num += 1
			else:
				features += [[sum(dist)/num, mean_angles(angl)]]
				dist, angl, num = [c_l], [normalize_angle(radians(k/2) - np.pi/2)], 1.0
	features += [[sum(dist) / num, mean_angles(angl)]]
	return features


def vic_f(current_state, u_t, Q, args=None, do_noise=True):
	x, y, theta = current_state[0], current_state[1], current_state[2]
	vel = u_t[0]
	alpha = u_t[1]
	a = args[1]
	delta_t = args[0]
	b = args[2]
	L = args[3]
	H = args[4]

	res = np.asarray([
		x + delta_t*(v*cos(theta)) - (vel/L)*tan(theta)*(a*sin(theta) + b*cos(theta)),
		y + delta_t*(v*sin(theta)) + (vel/L)*tan(theta)*(a*cos(theta) - b*sin(theta)),
		normalize_angle(theta + delta_t*(vel/L)*tan(alpha))
	]).T[0]

	# if norm(res[:2] - current_state[:2]) > 10.00:
	# 	print("Current u :" , u_t)
	# 	print("Current state: ", current_state)
	# 	print(res)
	# 	raise Exception

	if do_noise:
		noise = np.random.normal(0, Q.diagonal(), 3)
		res += noise

	current_state[:3] = res
	return current_state


np.set_printoptions(precision=3, suppress=True)

# Initialize the covariance matrices
ro_x = 0.1
ro_y = 0.1
ro_theta = 0.001

ro_range = 25
ro_bearing = 0.001

init_ro_x = 5
init_ro_y = 5
init_ro_theta = .1

Q, R, init_cov = generate_covariance_matrices(ro_x, ro_y, ro_theta, ro_range, ro_bearing, init_ro_x, init_ro_y, init_ro_theta)

Q_inflation = [2.5, 2.5, 2.5]
R_inflation = [2.5, 2.5]

# initialize additional arguments for the transition functions
(w, delta_t) = (1, 1)

# initialize additional information
number_of_ensemble_members = 100

folder = "../output/slam/victoria/"

# set up the Ensemble Kalman filter to use
EnsKFSlam_UA = EnsKFSlamUA(state=np.asarray(np.asarray([0.0, 0.0, 0.0])),
						N=number_of_ensemble_members,
						dim_state=3,
						num_landmarks=0,
						dim_z=2,
						Q=Q*Q_inflation,
						R=R*R_inflation,
						init_cov=None)

laser_mat = loadmat("../data/dataset_victoria_park/original_MATLAB_dataset/aa3_lsr2")
drive = loadmat("../data/dataset_victoria_park/original_MATLAB_dataset/aa3_dr")

steering = drive['steering']
vel = drive['speed']
driving_time_stamps = drive['time']

laser = laser_mat['LASER']
laser_recording_times = laser_mat['TLsr']

len_drive_information = len(drive['time'])

L, b, a, H_vic = 2.83, 0.5, 3.78, 0.76

current_laser_index = 0

ens_prediction = [EnsKFSlam_UA.state]

# iterate over driving information
for i in range(len_drive_information):

	if i % 250 == 0:
		print("--> Doing step %d" % i)

	v = vel[i]
	alpha = normalize_angle(steering[i])
	u_t = [v, alpha]

	timestamp = driving_time_stamps[i]

	# dirty hack!
	if i > 0:
		# Divide by thounds - the timestamp is in milliseconds
		# and the velocity is in m/sec
		delta_t = abs(timestamp - driving_time_stamps[i-1])/1000
	else:
		continue

	# print("Pre-predict: ", EnsKFSlam_UA.state)
	EnsKFSlam_UA.predict(vic_f, u_t, f_args=[delta_t, a, b, L, H_vic], landmark_white_noise=0.0)
	# print("After-predict: ", EnsKFSlam_UA.state)
	# print("-----")

	# while current_laser_index < len(laser_recording_times) and laser_recording_times[current_laser_index] <= timestamp:
	# 	meas = np.asarray(compute_measurements(laser[current_laser_index]))
	#
	# 	EnsKFSlam_UA.full_update(meas, h, H, threshold=8200)
	#
	# 	current_laser_index += 1

	# if norm(EnsKFSlam_UA.state[:2] - ens_prediction[-1][:2]) > 10.00 or i == 1200:
	# 	print("Current u :" , u_t)
	# 	print("Current state: ", ens_prediction[-1][:3])
	# 	print(EnsKFSlam_UA.state[:3])
	# 	print("History:")
	# 	for j in range(2,100):
	# 		print(ens_prediction[-j][:3])
	#
	# 	raise Exception


	ens_prediction += [EnsKFSlam_UA.state]

	if i % 250 == 0:

		max_x = max([max(elem) for elem in ens_prediction]) + 1000
		max_y = max([max(elem) for elem in ens_prediction]) + 1000

		min_x = min([min(elem) for elem in ens_prediction]) - 1000
		min_y = min([min(elem) for elem in ens_prediction]) - 1000

		plot_victoria("Performance of EnsSlam on Victoria", folder + "enskf_slam_victoria_progress_%d.pdf" % i, ens_prediction, min_x, max_x, min_y, max_y)


max_x = max([elem[0] for elem in ens_prediction])
max_y = max([elem[1] for elem in ens_prediction])
max_theta = max([elem[2] for elem in ens_prediction])

min_x = min([elem[0] for elem in ens_prediction])
min_y = min([elem[1] for elem in ens_prediction])
min_theta = min([elem[2] for elem in ens_prediction])


###############################################################################################################

print("Start printing")

plot_victoria("Performance of EnsSlam on Victoria", folder + "enskf_slam_victoria.pdf", ens_prediction, min_x, max_x, min_y, max_y)

exit()
########################################################################################################################

odo_file = "../data/dataset_victoria_park/victoria_park_dataset_ODO.txt"
odo = []

with open(odo_file, "r") as r_file:
	for line in r_file.readlines():
		odo += [[np.double(elem) for elem in line.split(" ")]]

##########

laser_file = "../data/dataset_victoria_park/victoria_park_dataset_LASER_.txt"
laser = []

with open(laser_file, "r") as r_file:
	for line in r_file.readlines():
		laser += [[np.double(elem) for elem in line.rstrip().split(" ")]]

len_laser = len(laser[0])
len_odo = len(odo[0])

for elem in laser:
	if len(elem) != len_laser:
		raise Exception

for elem in odo:
	if len(elem) != len_odo:
		raise Exception

number_of_steps = len(laser)

print("Found data for %d steps in files." % number_of_steps)

scans = [[i, laser[0][i]] for i in range(len(laser[0])) if laser[0][i] != 0.0]


def compute_next_position_according_to_odo(current_state, odo, Q, args=None, do_noise=True):
	current_state[:3] += odo
	if do_noise:
		noise = np.random.normal(0, Q.diagonal(), 3)
		current_state[:3] += odo
	return current_state


def compute_feature(scans):
	num_raw_scans = len(scans)
	feature_scans = []
	angle, dist = scans[0][0], scans[0][1]
	num = 1.0
	for i in range(1, num_raw_scans):
		if scans[i][0] == scans[i-1][0] + 1 and abs(scans[i][1] - scans[i-1][1]) < 1.0:
			angle += scans[i][0]
			dist += scans[i][1]
			num += 1.0
		else:
			# if not the case, previous recording list up to one feature. calc mid
			feature_scans += [[dist/num, normalize_angle(radians(angle/(2*num)) - np.pi/2)]]
			# reset counters
			angle = scans[i][0]
			dist = scans[i][1]
			num = 1.0
	feature_scans += [[dist / num, normalize_angle(radians(angle / (2*num)) - np.pi/2)]]
	return np.asarray(feature_scans)


all_scans = [compute_feature([[i, laser[j][i]] for i in range(len(laser[0])) if laser[j][i] != 0.0]) for j in range(len(laser))]


plot_result, plot_prediction, plot_update = True, False, False



start_time = time()





ens_prediction = [EnsKFSlam_UA.state]
###############################################################################################################

for i in range(number_of_steps):

	if i % 100 == 0:
		print("--> Doing step %d" % i)
	u = odo[i]
	z = all_scans[i]

	EnsKFSlam_UA.predict(compute_next_position_according_to_odo, u, f_args=None, landmark_white_noise=0.0)

	# if plot_prediction:
	# 	plot_final_slam_result("Progress for localization using an ensemble Kalman filter",
	# 						folder + "ens_progress/step_%d_pred.png" % i, position_at_i[:i + 1],
	# 						ens_prediction[:i] + [EnsKFSlam.state],
	# 						landmarks, x_min, x_max, None, EnsKFSlam.ensemble, do_particles=False,
	# 						do_ensemble=True,
	# 						path=True, show=False, save=True, slam=False)


	# for j in range(len(z)):
	# 	mes, according_landmark = z[j], landmarks[j]
	# 	if mes is None:
	# 		continue
	# 	EnsKFSlam.update(mes, h, j, h_args=according_landmark)
	#

	ens_prediction += [EnsKFSlam_UA.state]

	# if plot_update:
	# 	plot_final_slam_result("Progress for localization using an ensemble Kalman filter",
	# 						folder + "ens_progress/step_%d_up.png" % i, position_at_i[:i + 1],
	# 						ens_prediction[:i + 1],
	# 						landmarks, x_min, x_max, None, EnsKFSlam.ensemble, do_particles=False,
	# 						do_ensemble=True,
	# 						path=True, show=False, save=True, slam=True)


max_x = max([elem[0] for elem in ens_prediction])
max_y = max([elem[1] for elem in ens_prediction])
max_theta = max([elem[2] for elem in ens_prediction])

min_x = min([elem[0] for elem in ens_prediction])
min_y = min([elem[1] for elem in ens_prediction])
min_theta = min([elem[2] for elem in ens_prediction])


###############################################################################################################

if plot_result:
	plot_victoria("Performance of EnsSlam on Victoria", folder + "enskf_slam_victoria.pdf", ens_prediction, min_x, max_x, min_y, max_y)
